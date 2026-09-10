"""
MIRA — Analysis Engine
======================

Центральный модуль анализа одного рабочего сообщения.

Задача AnalysisEngine:
получить MessageEnvelope и превратить его в AnalysisResult,
который затем смогут использовать:

    ProjectStateEngine
        ↓
    история проекта
        ↓
    ReminderService
        ↓
    интерфейс / Telegram

ВАЖНО
-----
AnalysisEngine остаётся независимым от конкретной языковой модели.

Сейчас поддерживается гибридный режим:
- pretrained semantic analyzer (например, RuBERT) определяет смысловой intent;
- rule-based слой сохраняет детерминированное извлечение сроков, финансов,
  задач, требований, событий и safety-признаков;
- Context / ProjectState слой связывает анализ с историей проекта.

Позже тот же контракт позволяет подключить:
- multi-label pretrained модель;
- LLM для сложных требований/договорённостей;
- RAG для поиска предыдущих договорённостей;
- отдельный ProjectResolver;
- анализ вложений.

Ключевой принцип:
AnalysisEngine НЕ должен молча придумывать факты.
Если проект, срок, смысл короткого ответа или договорённость
не определены достаточно уверенно — ставится requires_human=True.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol, Tuple, Any

from .analysis_schema_v2 import (
    Agreement,
    AnalysisResult,
    ConflictAssessment,
    ConflictLevel,
    ContactRole,
    Decision,
    DecisionStatus,
    FinancialIssue,
    IntentItem,
    IntentType,
    MessageAnalysis,
    MessageContext,
    MessageEnvelope,
    MessageEvent,
    MessageEventType,
    PriorityLevel,
    ProjectReference,
    Requirement,
    RequirementStatus,
    Speaker,
    TaskItem,
    TaskStatus,
)


# ---------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ КОНТРАКТЫ
# ---------------------------------------------------------------------

class ExternalLanguageAnalyzer(Protocol):
    """
    Интерфейс для будущего RuBERT/LLM слоя.

    AnalysisEngine может использовать внешний анализатор,
    но не обязан зависеть от конкретной модели.
    """

    def analyze(
        self,
        message: MessageEnvelope,
    ) -> MessageAnalysis:
        ...




class ProjectContextProvider(Protocol):
    """
    Источник текущего состояния и истории проекта.

    Этому контракту уже соответствует ProjectStateEngine:
        get_state(project_id)
        get_history(project_id)

    Благодаря Protocol AnalysisEngine не зависит жёстко
    от конкретной реализации ProjectStateEngine.
    Позже вместо in-memory движка здесь сможет быть Repository/DB.
    """

    def get_state(self, project_id: str) -> Any:
        ...

    def get_history(self, project_id: str) -> List[Any]:
        ...


class DialogueContextProvider(Protocol):
    """
    Источник полной истории сообщений.

    Этому контракту соответствуют:
    - DialogueStore;
    - SQLiteDialogueStore.

    AnalysisEngine получает не всю историю целиком, а компактный
    DialogueContextBundle через build_context().
    """

    def build_context(
        self,
        dialog_id: str,
        current_message_id: Optional[str] = None,
        recent_limit: int = 20,
        include_reply_target: bool = True,
    ) -> Any:
        ...


@dataclass
class AnalysisEngineConfig:
    """
    Настройки первой версии AnalysisEngine.
    """

    default_confidence: float = 0.80

    # Если встречается срок, заданный другой стороной,
    # считаем его неподтверждённым пользователем,
    # пока в истории нет явного подтверждения.
    external_deadline_requires_confirmation: bool = True

    # Если короткий ответ пользователя нельзя связать с контекстом,
    # не интерпретируем его автоматически.
    short_reply_requires_context: bool = True

    # Сколько недавних сообщений брать из DialogueStore.
    # Это НЕ ограничение полной истории, а только context window.
    dialogue_recent_limit: int = 20

    # Уверенность для intent, восстановленного через явный reply_to.
    reply_context_confidence: float = 0.85

    # Уверенность для intent, восстановленного только по недавнему диалогу.
    recent_context_confidence: float = 0.65


# ---------------------------------------------------------------------
# ANALYSIS ENGINE
# ---------------------------------------------------------------------

class AnalysisEngine:
    """
    Центральный координатор анализа.

    Режимы:

    1. Только rule-based:
       если external_analyzer не передан.

    2. Hybrid:
       rule-based анализ выполняется ВСЕГДА;
       external_analyzer добавляет семантическое понимание текста;
       затем результаты объединяются и проходят через
       ProjectState/context/safety слой.

    Важно: pretrained intent-модель не заменяет извлечение задач,
    требований, сроков и финансов. Она является одним из слоёв MIRA.
    """

    def __init__(
        self,
        config: Optional[AnalysisEngineConfig] = None,
        external_analyzer: Optional[ExternalLanguageAnalyzer] = None,
        context_provider: Optional[ProjectContextProvider] = None,
        dialogue_context_provider: Optional[DialogueContextProvider] = None,
    ) -> None:
        self.config = config or AnalysisEngineConfig()
        self.external_analyzer = external_analyzer

        # Обычно сюда передаётся ProjectStateEngine.
        #
        # AnalysisEngine только ЧИТАЕТ состояние/историю.
        # Изменяет проект по-прежнему ProjectStateEngine.apply().
        self.context_provider = context_provider

        # Полная история сообщений хранится отдельно от ProjectState.
        # Обычно сюда передаётся DialogueStore или SQLiteDialogueStore.
        self.dialogue_context_provider = dialogue_context_provider

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def analyze(
        self,
        message: MessageEnvelope,
    ) -> AnalysisResult:
        """
        Анализирует одно сообщение и возвращает AnalysisResult.
        """

        # Rule-based слой выполняется всегда: он отвечает за
        # безопасные детерминированные извлечения.
        rule_analysis = self._analyze_rules(message)

        if self.external_analyzer is not None:
            external_analysis = self.external_analyzer.analyze(message)
            analysis = self._merge_analyses(
                rule_analysis=rule_analysis,
                external_analysis=external_analysis,
            )
        else:
            analysis = rule_analysis

        # Явные детерминированные сигналы не должны загрязняться
        # очевидно противоречащим одиночным прогнозом RuBERT.
        self._apply_deterministic_intent_precedence(
            message=message,
            analysis=analysis,
        )

        # -------------------------------------------------------------
        # КОНТЕКСТ ДИАЛОГА
        # -------------------------------------------------------------
        # Для зависимых реплик вроде:
        #   "Проверю."
        #   "Вы смотрели?"
        #   "А по этому?"
        #
        # одиночная pretrained-классификация может быть ошибочной.
        # Если DialogueStore доступен, пытаемся восстановить тему через
        # reply_to или ближайшее содержательное сообщение.
        self._enrich_with_dialogue_context(
            message=message,
            analysis=analysis,
        )

        # -------------------------------------------------------------
        # КОНТЕКСТ ПРОЕКТА
        # -------------------------------------------------------------
        # Здесь одиночный анализ сообщения связывается с тем,
        # что MIRA уже знает о проекте.
        #
        # Например:
        #   было: "шкаф 350 мм"
        #   пришло: "шкаф 250 вместо 350"
        #
        # Без контекста мы видим только новую цифру.
        # С контекстом можем найти старое requirement_id и заполнить
        # supersedes_requirement_id.
        self._enrich_with_project_context(
            message=message,
            analysis=analysis,
        )

        # После любого анализатора применяем общие ограничения.
        self._apply_context_safety(message, analysis)
        self._apply_deadline_safety(message, analysis)

        affected_project_ids = self._collect_affected_project_ids(
            message,
            analysis,
        )

        return AnalysisResult(
            message=message,
            analysis=analysis,
            affected_project_ids=affected_project_ids,
        )

    # -----------------------------------------------------------------
    # RULE-BASED MVP
    # -----------------------------------------------------------------

    def _analyze_rules(
        self,
        message: MessageEnvelope,
    ) -> MessageAnalysis:
        """
        Минимальный анализатор без ML.

        Его цель — не добиться высокой NLP-точности,
        а дать MIRA рабочий pipeline уже сейчас.
        """

        text = (message.text or "").strip()
        text_lower = text.lower()

        projects = list(message.projects)

        intents = self._detect_intents(text_lower, projects)
        requirements = self._extract_requirements(
            message,
            text,
            projects,
        )
        tasks = self._extract_tasks(
            message,
            text,
            projects,
        )
        agreements = self._extract_agreements(
            message,
            text,
            projects,
        )
        decisions = self._extract_decisions(
            message,
            text,
            projects,
        )
        financial_issues = self._extract_financial_issues(
            message,
            text,
            projects,
        )
        events = self._extract_events(
            message,
            text,
            projects,
        )

        conflict = self._detect_conflict(
            text_lower,
        )

        urgency = self._detect_urgency(
            text_lower,
        )

        analysis = MessageAnalysis(
            message_id=message.message_id,
            projects=projects,
            intents=intents,
            requirements=requirements,
            tasks=tasks,
            decisions=decisions,
            agreements=agreements,
            financial_issues=financial_issues,
            events=events,
            conflict=conflict,
            urgency=urgency,
            confidence=self.config.default_confidence,
        )

        # Если проект не определён — это важная причина
        # передать решение человеку.
        if not self._has_resolved_project(projects):
            analysis.requires_human = True
            analysis.human_reason = (
                "Не удалось однозначно определить проект."
            )

        return analysis

    # -----------------------------------------------------------------
    # HYBRID MERGE
    # -----------------------------------------------------------------

    def _merge_analyses(
        self,
        rule_analysis: MessageAnalysis,
        external_analysis: MessageAnalysis,
    ) -> MessageAnalysis:
        """Объединяет rule-based и pretrained/LLM анализ."""

        if external_analysis.message_id != rule_analysis.message_id:
            raise ValueError(
                "external_analyzer вернул MessageAnalysis для другого message_id."
            )

        projects = self._merge_projects(
            rule_analysis.projects, external_analysis.projects
        )
        intents = self._merge_intents(
            rule_analysis.intents, external_analysis.intents
        )

        tasks = self._merge_items_by_id(
            rule_analysis.tasks, external_analysis.tasks, "task_id"
        )
        requirements = self._merge_items_by_id(
            rule_analysis.requirements, external_analysis.requirements,
            "requirement_id"
        )
        decisions = self._merge_items_by_id(
            rule_analysis.decisions, external_analysis.decisions, "decision_id"
        )
        agreements = self._merge_items_by_id(
            rule_analysis.agreements, external_analysis.agreements, "agreement_id"
        )
        events = self._merge_items_by_id(
            rule_analysis.events, external_analysis.events, "event_id"
        )
        financial_issues = self._merge_financial_issues(
            rule_analysis.financial_issues, external_analysis.financial_issues
        )
        conflict = self._merge_conflict(
            rule_analysis.conflict, external_analysis.conflict
        )
        urgency = self._max_priority(
            rule_analysis.urgency, external_analysis.urgency
        )

        requires_human = (
            rule_analysis.requires_human
            or external_analysis.requires_human
            or conflict.requires_human
        )

        human_reason = self._merge_reason_text(
            rule_analysis.human_reason, external_analysis.human_reason
        )
        if conflict.requires_human and conflict.reason:
            human_reason = self._merge_reason_text(
                human_reason, conflict.reason
            )

        return MessageAnalysis(
            message_id=rule_analysis.message_id,
            projects=projects,
            intents=intents,
            tasks=tasks,
            requirements=requirements,
            decisions=decisions,
            agreements=agreements,
            financial_issues=financial_issues,
            conflict=conflict,
            events=events,
            urgency=urgency,
            requires_human=requires_human,
            human_reason=human_reason,
            recommendation=(
                external_analysis.recommendation or rule_analysis.recommendation
            ),
            proposed_response=(
                external_analysis.proposed_response or rule_analysis.proposed_response
            ),
            confidence=max(
                rule_analysis.confidence, external_analysis.confidence
            ),
        )

    @staticmethod
    def _merge_projects(
        left: List[ProjectReference],
        right: List[ProjectReference],
    ) -> List[ProjectReference]:
        result: List[ProjectReference] = []
        by_key: Dict[Tuple[Optional[str], Optional[str]], ProjectReference] = {}

        for item in list(left) + list(right):
            key = (item.project_id, item.project_name)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = item
                result.append(item)
                continue
            if item.confidence > existing.confidence:
                index = result.index(existing)
                result[index] = item
                by_key[key] = item

        return result

    @staticmethod
    def _merge_intents(
        left: List[IntentItem],
        right: List[IntentItem],
    ) -> List[IntentItem]:
        """Объединяет intent по типу, не теряя вторичный смысл."""
        result: List[IntentItem] = []
        by_type: Dict[IntentType, IntentItem] = {}

        for item in list(left) + list(right):
            existing = by_type.get(item.intent)
            if existing is None:
                by_type[item.intent] = item
                result.append(item)
                continue

            if item.confidence > existing.confidence:
                if item.project is None and existing.project is not None:
                    item.project = existing.project
                index = result.index(existing)
                result[index] = item
                by_type[item.intent] = item
            elif existing.project is None and item.project is not None:
                existing.project = item.project

        return result

    @staticmethod
    def _merge_items_by_id(
        left: List[Any],
        right: List[Any],
        id_attribute: str,
    ) -> List[Any]:
        result = list(left)
        known_ids = {
            getattr(item, id_attribute, None)
            for item in result
            if getattr(item, id_attribute, None)
        }
        for item in right:
            item_id = getattr(item, id_attribute, None)
            if item_id and item_id in known_ids:
                continue
            result.append(item)
            if item_id:
                known_ids.add(item_id)
        return result

    @staticmethod
    def _merge_financial_issues(
        left: List[FinancialIssue],
        right: List[FinancialIssue],
    ) -> List[FinancialIssue]:
        result: List[FinancialIssue] = []
        seen = set()
        for item in list(left) + list(right):
            project_id = (
                item.project.project_id if item.project is not None else None
            )
            key = (item.topic, item.amount, item.currency, item.status, project_id)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _merge_conflict(
        left: ConflictAssessment,
        right: ConflictAssessment,
    ) -> ConflictAssessment:
        level_rank = {
            ConflictLevel.NONE: 0,
            ConflictLevel.LOW: 1,
            ConflictLevel.MEDIUM: 2,
            ConflictLevel.HIGH: 3,
        }
        level = (
            left.level
            if level_rank.get(left.level, 0) >= level_rank.get(right.level, 0)
            else right.level
        )
        candidates = [item for item in (left, right) if item.reason]
        reason = (
            max(candidates, key=lambda item: item.confidence).reason
            if candidates else None
        )
        return ConflictAssessment(
            message_conflict=left.message_conflict or right.message_conflict,
            dialog_conflict=left.dialog_conflict or right.dialog_conflict,
            level=level,
            confidence=max(left.confidence, right.confidence),
            reason=reason,
            requires_human=left.requires_human or right.requires_human,
        )

    @staticmethod
    def _max_priority(
        left: PriorityLevel,
        right: PriorityLevel,
    ) -> PriorityLevel:
        rank = {
            PriorityLevel.UNKNOWN: -1,
            PriorityLevel.LOW: 0,
            PriorityLevel.NORMAL: 1,
            PriorityLevel.HIGH: 2,
            PriorityLevel.CRITICAL: 3,
        }
        return left if rank.get(left, -1) >= rank.get(right, -1) else right

    @staticmethod
    def _merge_reason_text(
        left: Optional[str],
        right: Optional[str],
    ) -> Optional[str]:
        parts: List[str] = []
        for value in (left, right):
            if not value:
                continue
            cleaned = value.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
        return " ".join(parts) if parts else None


    # -----------------------------------------------------------------
    # INTENT
    # -----------------------------------------------------------------

    def _detect_intents(
        self,
        text_lower: str,
        projects: List[ProjectReference],
    ) -> List[IntentItem]:
        """
        Очень простой multi-label intent detector.

        В отличие от старой LSTM, здесь сообщение может иметь
        несколько intent одновременно.
        """

        intents: List[IntentItem] = []

        rules = [
            (
                IntentType.EDITS,
                [
                    "поменя",
                    "замени",
                    "исправ",
                    "правк",
                    "сделай шире",
                    "сделай уже",
                    "добавь",
                    "убери",
                    "вместо",
                ],
            ),
            (
                IntentType.DEADLINE,
                [
                    "сегодня",
                    "завтра",
                    "до ",
                    "к ",
                    "срок",
                    "успе",
                    "когда",
                ],
            ),
            (
                IntentType.COST,
                [
                    "стоим",
                    "цена",
                    "сколько стоит",
                    "сколько будет стоить",
                    "стоить",
                    "сколько с меня",
                    "доплат",
                ],
            ),
            (
                IntentType.PAYMENT,
                [
                    "оплата",
                    "оплатил",
                    "оплатила",
                    "оплачено",
                    "перевод",
                    "поступлен",
                    "задолж",
                    "выплат",
                    "зарплат",
                    "счёт",
                    "счет",
                ],
            ),
            (
                IntentType.RENDER,
                [
                    "рендер",
                    "пререндер",
                    "ракурс",
                    "визуализац",
                ],
            ),
            (
                IntentType.MATERIALS,
                [
                    "материал",
                    "дерево",
                    "шпон",
                    "стекло",
                    "хром",
                    "плитк",
                ],
            ),
            (
                IntentType.TECHNICAL_QUESTION,
                [
                    "мм",
                    "см",
                    "метр",
                    "размер",
                    "глубин",
                    "высот",
                    "ширин",
                ],
            ),
            (
                IntentType.APPROVAL,
                [
                    "ок",
                    "утверждаем",
                    "подходит",
                    "согласовано",
                    "одобр",
                ],
            ),
            (
                IntentType.REJECTION,
                [
                    "не подходит",
                    "не утверждаем",
                    "отмен",
                    "не делаем",
                ],
            ),
            (
                IntentType.RESULT_DELIVERY,
                [
                    "отправил",
                    "отправила",
                    "готово",
                    "финал",
                ],
            ),
        ]

        for intent_type, markers in rules:
            matched = any(
                self._intent_marker_matches(
                    text_lower,
                    marker,
                )
                for marker in markers
            )

            # Отдельно учитываем формы глагола "менять":
            # меняем / меняю / меняйте / менять / меняли ...
            #
            # Нельзя просто добавить маркер "меня", иначе фраза
            # "у меня не было перевода" ошибочно станет EDITS.
            if (
                intent_type == IntentType.EDITS
                and self._edit_action_matches(text_lower)
            ):
                matched = True

            if matched:
                intents.append(
                    IntentItem(
                        intent=intent_type,
                        confidence=0.75,
                        project=self._single_project_or_none(projects),
                    )
                )

        return self._unique_intents(intents)

    @staticmethod
    def _edit_action_matches(
        text_lower: str,
    ) -> bool:
        """
        Безопасно распознаёт формы глагола "менять",
        не путая их с местоимением "меня".

        Совпадает:
            "меняем плитку"
            "я меняю цвет"
            "менять материал"
            "меняйте зеркало"
            "вчера меняли шпон"

        Не совпадает:
            "у меня нет оплаты"
        """

        return bool(
            re.search(
                r"(?<![а-яёa-z])"
                r"меня(?:ть|ю|ешь|ет|ем|ете|ют|й|йте|л|ла|ли)"
                r"(?![а-яёa-z])",
                text_lower,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _intent_marker_matches(
        text_lower: str,
        marker: str,
    ) -> bool:
        """
        Безопасное сопоставление rule-маркеров intent.

        Раньше простое `marker in text` давало ложные совпадения:
            "см"  находилось внутри "смотрели"
            "ок"  находилось внутри "срок"

        Для коротких самостоятельных токенов требуем границу слова.
        При этом единицы после цифры продолжают работать:
            "25 см"
            "250мм"
        """

        short_tokens = {
            "мм",
            "см",
            "ок",
        }

        if marker in short_tokens:
            pattern = (
                r"(?<![а-яёa-z])"
                + re.escape(marker)
                + r"(?![а-яёa-z])"
            )

            return bool(
                re.search(
                    pattern,
                    text_lower,
                    flags=re.IGNORECASE,
                )
            )

        return marker in text_lower


    # -----------------------------------------------------------------
    # REQUIREMENTS
    # -----------------------------------------------------------------

    def _extract_requirements(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[Requirement]:
        """
        Извлекает простые требования, в том числе несколько замен
        вида "X 250 мм вместо 350 мм" в одном сообщении.

        Это по-прежнему rule-based MVP: сложные формулировки позже
        должен разбирать LLM/entity extractor.
        """

        result: List[Requirement] = []
        project = self._single_project_or_none(projects)

        # Ищем ВСЕ конструкции "новое значение вместо старого".
        # Prefix ограничен текущим предложением, чтобы в entity не
        # попадал хвост предыдущей правки.
        replacement_pattern = re.compile(
            r"(?P<prefix>[^.!?;]{0,80}?)"
            r"(?P<new>\d+(?:[.,]\d+)?)\s*(?P<new_unit>мм|см|м)?"
            r"\s+вместо\s+"
            r"(?P<old>\d+(?:[.,]\d+)?)\s*(?P<old_unit>мм|см|м)?",
            re.IGNORECASE,
        )
        replacements = list(replacement_pattern.finditer(text))

        for idx, replacement in enumerate(replacements, start=1):
            new_value = replacement.group("new")
            new_unit = (
                replacement.group("new_unit")
                or replacement.group("old_unit")
            )

            prefix = replacement.group("prefix").strip()
            prefix = re.sub(
                r"\b(теперь|делаем|сделать|будет|должен быть|должна быть)\b",
                " ",
                prefix,
                flags=re.IGNORECASE,
            )
            prefix_words = re.findall(r"[а-яёa-z0-9_-]+", prefix.lower())
            entity = prefix_words[-1] if prefix_words else None

            # Сохраняем прежний id для одиночной замены, чтобы не ломать
            # уже существующие тесты/историю. Для нескольких — уникальные ids.
            if len(replacements) == 1:
                requirement_id = f"req_{message.message_id}_replacement"
            else:
                requirement_id = f"req_{message.message_id}_replacement_{idx}"

            result.append(
                Requirement(
                    requirement_id=requirement_id,
                    description=(
                        (f"{entity}: " if entity else "")
                        + f"{new_value}"
                        + (f" {new_unit}" if new_unit else "")
                    ),
                    project=project,
                    source_message_id=message.message_id,
                    status=RequirementStatus.ACTIVE,
                    confidence=0.80,
                    entity=entity,
                    value=new_value,
                    unit=new_unit,
                    requires_human_confirmation=True,
                )
            )

        # Простые размеры, не входящие в replacement-конструкции.
        dimension_matches = re.finditer(
            r"\b([А-Яа-яA-Za-zЁё]+)\s+"
            r"(\d+(?:[.,]\d+)?)\s*(мм|см|м)\b",
            text,
        )

        replacement_spans = [item.span() for item in replacements]

        for idx, match in enumerate(dimension_matches, start=1):
            # Любое измерение, пересекающееся с конструкцией X вместо Y,
            # уже представлено replacement-requirement и повторно не создаётся.
            if any(
                match.start() >= span_start and match.end() <= span_end
                for span_start, span_end in replacement_spans
            ):
                continue

            entity = match.group(1)
            value = match.group(2)
            unit = match.group(3)

            result.append(
                Requirement(
                    requirement_id=f"req_{message.message_id}_dim_{idx}",
                    description=f"{entity}: {value} {unit}",
                    project=project,
                    source_message_id=message.message_id,
                    status=RequirementStatus.ACTIVE,
                    confidence=0.78,
                    entity=entity,
                    value=value,
                    unit=unit,
                )
            )

        return result

    # -----------------------------------------------------------------
    # TASKS
    # -----------------------------------------------------------------

    def _extract_tasks(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[TaskItem]:
        """
        Извлекает простые рабочие задачи.

        Важнее всего сейчас:
        - текст задачи;
        - project;
        - deadline;
        - требуется ли подтверждение пользователя.
        """

        result: List[TaskItem] = []
        project = self._single_project_or_none(projects)

        task_markers = [
            "сделай",
            "замени",
            "поменя",
            "исправ",
            "добавь",
            "убери",
            "пришли",
            "отправ",
            "подготов",
            "проверь",
        ]

        if not any(marker in text.lower() for marker in task_markers):
            return result

        deadline_text, deadline_iso = self._extract_deadline(
            text=text,
            message_time=message.timestamp,
        )

        task = TaskItem(
            task_id=f"task_{message.message_id}_1",
            description=text.strip(),
            project=project,
            deadline_text=deadline_text,
            deadline_iso=deadline_iso,
            deadline_confirmed_by_user=(
                message.speaker == Speaker.USER
            ),
            source_message_id=message.message_id,
            status=TaskStatus.NEW,
            requires_confirmation=(
                bool(deadline_iso)
                and message.speaker != Speaker.USER
                and self.config.external_deadline_requires_confirmation
            ),
        )

        result.append(task)
        return result

    # -----------------------------------------------------------------
    # AGREEMENTS
    # -----------------------------------------------------------------

    def _extract_agreements(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[Agreement]:
        """
        Извлекает простые договорённости с явным согласованием.

        Пример:
            "В понедельник согласуем второй вариант."
        """

        text_lower = text.lower()

        agreement_markers = [
            "договорились",
            "согласуем",
            "согласовано",
            "утверждаем",
        ]

        if not any(marker in text_lower for marker in agreement_markers):
            return []

        project = self._single_project_or_none(projects)

        deadline_text, deadline_iso = self._extract_deadline(
            text,
            message.timestamp,
        )

        return [
            Agreement(
                agreement_id=f"agr_{message.message_id}_1",
                description=text.strip(),
                project=project,
                active=True,
                deadline_text=deadline_text,
                deadline_iso=deadline_iso,
                reminder_required=bool(deadline_iso),
                source_message_ids=[message.message_id],
                confirmed_by=[message.contact_id]
                if message.contact_id
                else [],
            )
        ]

    # -----------------------------------------------------------------
    # DECISIONS
    # -----------------------------------------------------------------

    def _extract_decisions(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[Decision]:
        """
        Фиксирует явные решения пользователя.

        Особенно важно для коротких ответов:
            "Поняла"
            "Да"
            "Хорошо"
            "Сделаю"

        Они имеют смысл только если есть reply/context.
        """

        if message.speaker != Speaker.USER:
            return []

        normalized = self._normalize_short_reply_text(text)

        short_accepts = {
            "поняла",
            "понял",
            "да",
            "хорошо",
            "ок",
            "окей",
            "ясно",
            "сделаю",
            "принято",
        }

        if normalized not in short_accepts:
            return []

        project = self._single_project_or_none(projects)

        has_context = self._has_resolved_context(
            message.context,
        )

        return [
            Decision(
                decision_id=f"decision_{message.message_id}_1",
                description=(
                    "Пользователь подтвердил связанное сообщение"
                    if has_context
                    else "Короткий ответ без достаточного контекста"
                ),
                project=project,
                source_message_id=message.message_id,
                status=(
                    DecisionStatus.CONFIRMED
                    if has_context
                    else DecisionStatus.UNCLEAR
                ),
                confirmed_by_user=has_context,
                confidence=0.95 if has_context else 0.40,
                inferred_from_context=has_context,
            )
        ]

    # -----------------------------------------------------------------
    # FINANCIAL
    # -----------------------------------------------------------------

    def _extract_financial_issues(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[FinancialIssue]:
        """
        Минимальное извлечение финансового вопроса.
        """

        lower = text.lower()

        markers = [
            "стоимость",
            "сколько стоит",
            "сколько будет стоить",
            "стоить",
            "доплат",
            "входит в стоимость",
            "оплачивается отдельно",
            "сколько с меня",
            "цена",
            "оплата",
            "счёт",
            "счет",
            "перевод",
        ]

        if not any(marker in lower for marker in markers):
            return []

        project = self._single_project_or_none(projects)

        amount_match = re.search(
            r"\b(\d+(?:[ .,]\d{3})*(?:[.,]\d+)?)\s*"
            r"(руб(?:лей|ля|ль)?|₽|р\.?|usd|eur|доллар(?:ов|а)?|евро)?",
            text.lower(),
        )

        amount = None
        currency = None

        if amount_match:
            raw_amount = amount_match.group(1).replace(" ", "")
            raw_amount = raw_amount.replace(",", ".")
            try:
                amount = float(raw_amount)
            except ValueError:
                amount = None

            raw_currency = amount_match.group(2)
            if raw_currency:
                if raw_currency in {"₽", "р", "р.", "руб", "рубль", "рубля", "рублей"}:
                    currency = "RUB"
                elif raw_currency == "usd" or raw_currency.startswith("доллар"):
                    currency = "USD"
                elif raw_currency == "eur" or raw_currency == "евро":
                    currency = "EUR"

        return [
            FinancialIssue(
                detected=True,
                topic=text.strip(),
                amount=amount,
                currency=currency,
                status="unknown",
                project=project,
                requires_human=True,
            )
        ]

    # -----------------------------------------------------------------
    # EVENTS
    # -----------------------------------------------------------------

    def _extract_events(
        self,
        message: MessageEnvelope,
        text: str,
        projects: List[ProjectReference],
    ) -> List[MessageEvent]:
        """
        Выделяет события с конкретной датой/временем.

        Сейчас это пригодится прежде всего для ReminderService.
        """

        lower = text.lower()

        event_markers = [
            "созвон",
            "встреч",
            "обсудим",
            "согласуем",
        ]

        if not any(marker in lower for marker in event_markers):
            return []

        deadline_text, deadline_iso = self._extract_deadline(
            text,
            message.timestamp,
        )

        if not deadline_iso:
            return []

        project = self._single_project_or_none(projects)

        return [
            MessageEvent(
                event_id=f"event_{message.message_id}_1",
                event_type=MessageEventType.OTHER,
                project=project,
                source_message_id=message.message_id,
                description=text.strip(),
                scheduled_for_text=deadline_text,
                scheduled_for_iso=deadline_iso,
                reminder_required=True,
                confidence=0.75,
            )
        ]

    # -----------------------------------------------------------------
    # CONFLICT
    # -----------------------------------------------------------------

    def _detect_conflict(
        self,
        text_lower: str,
    ) -> ConflictAssessment:
        """
        Rule-based message_conflict.

        Это только локальный конфликт сообщения.
        Состояние конфликта всего диалога должно позже вычисляться
        через историю проекта/диалога.
        """

        high_markers = [
            "вы вообще",
            "мы не договаривались",
            "обещали",
            "почему вы",
            "сколько можно",
            "даже не отвечаете",
        ]

        medium_markers = [
            "не устраивает",
            "не подходит",
            "опять",
            "задерж",
            "когда уже",
        ]

        if any(marker in text_lower for marker in high_markers):
            return ConflictAssessment(
                message_conflict=True,
                level=ConflictLevel.HIGH,
                confidence=0.85,
                reason=(
                    "В сообщении обнаружены явные признаки "
                    "недовольства или претензии."
                ),
                requires_human=True,
            )

        if any(marker in text_lower for marker in medium_markers):
            return ConflictAssessment(
                message_conflict=True,
                level=ConflictLevel.MEDIUM,
                confidence=0.70,
                reason=(
                    "В сообщении есть возможные признаки "
                    "напряжённой ситуации."
                ),
                requires_human=True,
            )

        return ConflictAssessment(
            message_conflict=False,
            level=ConflictLevel.NONE,
            confidence=0.70,
        )

    # -----------------------------------------------------------------
    # URGENCY
    # -----------------------------------------------------------------

    @staticmethod
    def _detect_urgency(
        text_lower: str,
    ) -> PriorityLevel:
        """
        Минимальная оценка срочности.
        """

        high = [
            "срочно",
            "сегодня",
            "до вечера",
            "до обеда",
            "прямо сейчас",
        ]

        if any(marker in text_lower for marker in high):
            return PriorityLevel.HIGH

        medium = [
            "завтра",
            "на этой неделе",
            "в ближайшее время",
        ]

        if any(marker in text_lower for marker in medium):
            return PriorityLevel.NORMAL

        return PriorityLevel.NORMAL

    # -----------------------------------------------------------------
    # DEADLINE PARSER
    # -----------------------------------------------------------------

    def _extract_deadline(
        self,
        text: str,
        message_time: Optional[datetime],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Первый простой парсер дат.

        Поддерживает:
        - сегодня
        - завтра
        - время вида 18:00 / 18.00
        - "до 18"
        - "к 18"

        ВНИМАНИЕ:
        сложные выражения ("в следующий четверг после обеда")
        позже должен разбирать отдельный DateTimeExtractor/LLM.
        """

        lower = text.lower()
        base = message_time or datetime.now()

        target_date = None
        date_text = None

        if "сегодня" in lower:
            target_date = base.date()
            date_text = "сегодня"

        elif "завтра" in lower:
            target_date = (base + timedelta(days=1)).date()
            date_text = "завтра"

        # Ищем HH:MM / HH.MM
        time_match = re.search(
            r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b",
            lower,
        )

        hour = None
        minute = 0

        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
        else:
            # "до 18", "к 18"
            short_time = re.search(
                r"(?:до|к)\s+([01]?\d|2[0-3])\b",
                lower,
            )
            if short_time:
                hour = int(short_time.group(1))

        # Без даты сейчас не придумываем конкретный календарный день.
        if target_date is None:
            return None, None

        if hour is None:
            # Если есть только день, оставляем 09:00 техническим временем.
            hour = 9
            minute = 0

        due = datetime.combine(
            target_date,
            datetime.min.time(),
        ).replace(
            hour=hour,
            minute=minute,
        )

        # Сохраняем максимально близкую к исходнику формулировку.
        deadline_text = date_text
        if time_match:
            deadline_text += f" {time_match.group(0)}"
        elif hour is not None:
            if re.search(r"(?:до|к)\s+\d", lower):
                deadline_text += f" к {hour:02d}:00"

        return deadline_text, due.isoformat()

    # -----------------------------------------------------------------
    # DETERMINISTIC INTENT PRECEDENCE
    # -----------------------------------------------------------------

    def _apply_deterministic_intent_precedence(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> None:
        """
        Убирает только очевидное "загрязнение" hybrid-результата.

        Пример:
            "У меня в этом месяце не было перевода."

        Rules:
            PAYMENT

        RuBERT:
            MATERIALS

        Итог должен быть:
            PAYMENT

        Но если сообщение реально многосмысловое:
            "Оплату получила, плитку меняем на белую."

        Rules дают:
            PAYMENT + EDITS + MATERIALS

        и все эти intents сохраняются.

        То есть мы НЕ заменяем модель правилами.
        Мы подавляем только неподтверждённые rule-слоем явно чужие
        категории в сообщениях с сильным финансовым сигналом.
        """

        text = (message.text or "").strip()

        if not text:
            return

        projects = list(message.projects)

        rule_items = self._detect_intents(
            text.lower(),
            projects,
        )

        rule_types = {
            item.intent
            for item in rule_items
        }

        # Сильный финансовый сигнал должен подтверждаться не только
        # intent rule, но и отдельным FinancialIssue extractor.
        financial = self._extract_financial_issues(
            message,
            text,
            projects,
        )

        if not financial:
            return

        has_payment = (
            IntentType.PAYMENT in rule_types
        )
        has_cost = (
            IntentType.COST in rule_types
        )

        if not (has_payment or has_cost):
            return

        # Эти классы часто были уверенными ложными ответами RuBERT
        # на коротких/непохожих реальных финансовых сообщениях.
        # Если они действительно присутствуют в тексте, rule detector
        # их тоже найдёт — и тогда мы их сохраним.
        suppress_if_not_rule_supported = {
            IntentType.MATERIALS,
            IntentType.RENDER,
            IntentType.TECHNICAL_QUESTION,
            IntentType.EDITS,
        }

        cleaned: List[IntentItem] = []

        for item in analysis.intents:
            if (
                item.intent in suppress_if_not_rule_supported
                and item.intent not in rule_types
            ):
                continue

            cleaned.append(item)

        # На случай если merge по какой-то причине не сохранил
        # rule-based PAYMENT/COST, возвращаем их.
        existing_types = {
            item.intent
            for item in cleaned
        }

        for rule_item in rule_items:
            if (
                rule_item.intent in {
                    IntentType.PAYMENT,
                    IntentType.COST,
                }
                and rule_item.intent not in existing_types
            ):
                cleaned.append(rule_item)
                existing_types.add(
                    rule_item.intent
                )

        analysis.intents = self._unique_intents(
            cleaned
        )


    # -----------------------------------------------------------------
    # DIALOGUE CONTEXT ENRICHMENT
    # -----------------------------------------------------------------

    def _enrich_with_dialogue_context(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> None:
        """
        Восстанавливает смысл короткой/зависимой реплики через DialogueStore.

        ВАЖНО:
        текущий pretrained RuBERT обучен как single-message classifier.
        Мы НЕ склеиваем всю переписку в одну строку и не делаем вид,
        что модель уже обучена контексту.

        Вместо этого:
        1. определяем, является ли текущая реплика зависимой;
        2. ищем явный reply_to;
        3. если его нет — смотрим ближайшее содержательное сообщение;
        4. наследуем только ОДНО однозначное рабочее намерение;
        5. без явного reply_to оставляем requires_human=True.

        Это безопасный промежуточный слой до настоящей context-aware
        multi-label модели/LLM.
        """

        if self.dialogue_context_provider is None:
            return

        if not self._is_context_dependent_text(
            message.text
        ):
            return

        try:
            bundle = self.dialogue_context_provider.build_context(
                dialog_id=message.dialog_id,
                current_message_id=message.message_id,
                recent_limit=self.config.dialogue_recent_limit,
                include_reply_target=True,
            )
        except Exception:
            # История не должна ломать основной анализ.
            analysis.requires_human = True

            if not analysis.human_reason:
                analysis.human_reason = (
                    "Не удалось получить контекст диалога."
                )

            return

        reply_target = getattr(
            bundle,
            "reply_target",
            None,
        )

        # -------------------------------------------------------------
        # 1. Явный reply_to — самый надёжный источник.
        # -------------------------------------------------------------
        if reply_target is not None:
            inferred = self._infer_single_context_intent(
                reply_target
            )

            if inferred is not None:
                self._apply_context_inferred_intent(
                    current_message=message,
                    analysis=analysis,
                    inferred_intent=inferred,
                    source_message=reply_target,
                    confidence=self.config.reply_context_confidence,
                    explicit_reply=True,
                )

            return

        # -------------------------------------------------------------
        # 2. Без reply_to ищем ближайшее СОДЕРЖАТЕЛЬНОЕ сообщение.
        # -------------------------------------------------------------
        recent_messages = list(
            getattr(
                bundle,
                "recent_messages",
                [],
            )
        )

        for previous in reversed(
            recent_messages
        ):
            if previous.message_id == message.message_id:
                continue

            # "Проверю", "Да", "Вы смотрели?" сами не являются
            # надёжным источником темы — пропускаем их.
            if self._is_context_dependent_text(
                previous.text
            ):
                continue

            inferred = self._infer_single_context_intent(
                previous
            )

            if inferred is None:
                continue

            self._apply_context_inferred_intent(
                current_message=message,
                analysis=analysis,
                inferred_intent=inferred,
                source_message=previous,
                confidence=self.config.recent_context_confidence,
                explicit_reply=False,
            )

            return

    def _infer_single_context_intent(
        self,
        source_message: MessageEnvelope,
    ) -> Optional[IntentType]:
        """
        Определяет тему ОДНОГО исторического сообщения.

        Используем несколько сигналов:
        - rule intents;
        - financial detector;
        - pretrained semantic analyzer.

        Возвращаем intent только если после фильтрации тема однозначна.
        """

        text = (
            source_message.text
            or ""
        ).strip()

        if not text:
            return None

        projects = list(
            source_message.projects
        )

        candidates: List[IntentType] = []

        # -------------------------------------------------------------
        # 1. Детерминированная финансовая тема имеет приоритет.
        # -------------------------------------------------------------
        # Это важный safety-принцип:
        # если в историческом сообщении явно есть "перевод", "оплата",
        # "счёт" и т.п., ошибочный single-message RuBERT не должен
        # превращать PAYMENT в MATERIALS/RENDER и создавать ложную
        # "неоднозначность".
        financial = self._extract_financial_issues(
            source_message,
            text,
            projects,
        )

        if financial:
            lower = text.lower()

            if any(
                marker in lower
                for marker in [
                    "сколько",
                    "стоим",
                    "цена",
                    "доплат",
                ]
            ):
                return IntentType.COST

            return IntentType.PAYMENT

        # -------------------------------------------------------------
        # 2. Остальные rule-based semantic hints.
        # -------------------------------------------------------------
        for item in self._detect_intents(
            text.lower(),
            projects,
        ):
            candidates.append(
                item.intent
            )

        # -------------------------------------------------------------
        # 3. Pretrained semantic signal.
        # -------------------------------------------------------------
        if self.external_analyzer is not None:
            try:
                external = self.external_analyzer.analyze(
                    source_message
                )

                for item in external.intents:
                    candidates.append(
                        item.intent
                    )
            except Exception:
                # Историческое сообщение не должно ломать анализ текущего.
                pass

        # Generic labels не являются хорошей "темой" для наследования.
        ignored = {
            IntentType.UNKNOWN,
            IntentType.OTHER,
            IntentType.PRIMARY_CONTACT,
            IntentType.THANKS,
            IntentType.APPROVAL,
            IntentType.REJECTION,
        }

        unique: List[IntentType] = []

        for intent in candidates:
            if intent in ignored:
                continue

            if intent not in unique:
                unique.append(
                    intent
                )

        # Если несколько смыслов — не угадываем один.
        if len(unique) != 1:
            return None

        return unique[0]

    def _apply_context_inferred_intent(
        self,
        current_message: MessageEnvelope,
        analysis: MessageAnalysis,
        inferred_intent: IntentType,
        source_message: MessageEnvelope,
        confidence: float,
        explicit_reply: bool,
    ) -> None:
        """
        Применяет intent, восстановленный по контексту.

        Если у зависимой текущей реплики нет явного rule-based смысла,
        ошибочный single-message RuBERT intent заменяем контекстным.

        Если текущий текст содержит явные rule markers, ничего не стираем,
        а только добавляем контекстный смысл.
        """

        explicit_current = self._detect_intents(
            (current_message.text or "").lower(),
            list(current_message.projects),
        )

        project = self._single_project_or_none(
            current_message.projects
        )

        inferred_item = IntentItem(
            intent=inferred_intent,
            confidence=confidence,
            project=project,
            evidence=(
                "Контекст: "
                f"{source_message.message_id}: "
                f"{(source_message.text or '').strip()[:180]}"
            ),
        )

        if not explicit_current:
            # Для контекстно-зависимой реплики single-message
            # pretrained intent ненадёжен и может быть очень уверенно неверным.
            analysis.intents = [
                inferred_item
            ]
        else:
            analysis.intents = self._merge_intents(
                analysis.intents,
                [inferred_item],
            )

        # Без явного reply_to это пока эвристическое восстановление темы.
        if not explicit_reply:
            analysis.requires_human = True

            reason = (
                "Смысл сообщения восстановлен по недавнему контексту "
                "без явного reply_to."
            )

            analysis.human_reason = self._merge_reason_text(
                analysis.human_reason,
                reason,
            )

    @staticmethod
    def _is_context_dependent_text(
        value: Optional[str],
    ) -> bool:
        """
        Короткие рабочие реплики, смысл которых обычно находится
        в предыдущем сообщении.

        Это не список intent-классов, а сигнал:
        "не доверять одиночной классификации без контекста".
        """

        raw = (value or "").strip().lower()
        raw = re.sub(r"\s+", " ", raw)
        raw = re.sub(
            r"^\W+|\W+$",
            "",
            raw,
            flags=re.UNICODE,
        )

        exact = {
            "да",
            "нет",
            "ок",
            "окей",
            "хорошо",
            "поняла",
            "понял",
            "ясно",
            "принято",
            "сделаю",
            "проверю",
            "посмотрю",
            "уточню",
            "проверила",
            "проверил",
            "посмотрела",
            "посмотрел",
            "получилось",
            "готово",
            "вы смотрели",
            "вы посмотрели",
            "смотрели",
            "посмотрели",
            "ну что",
            "и что",
            "что там",
            "что по этому",
            "а по этому",
            "а по второму",
        }

        if raw in exact:
            return True

        # Очень короткий вопрос со ссылочным словом:
        # "А по нему?", "Что с этим?", "По второму?"
        if (
            len(raw.split()) <= 5
            and re.search(
                r"\b(это|этим|этому|нему|ней|второму|первому|там)\b",
                raw,
            )
        ):
            return True

        return False


    # -----------------------------------------------------------------
    # PROJECT CONTEXT ENRICHMENT
    # -----------------------------------------------------------------

    def _enrich_with_project_context(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> None:
        """
        Обогащает анализ текущим состоянием и историей проекта.

        Это первый слой "памяти" MIRA.

        Сейчас решает три особенно важных задачи:

        1. Находит, какое старое требование заменяется новым.
        2. Связывает задачу с requirement, созданным в том же сообщении.
        3. Для короткого ответа пользователя ("Поняла") пытается
           определить, какие requirements были подтверждены.

        Если контекст неоднозначен — ничего не угадываем.
        """

        if self.context_provider is None:
            return

        resolved_projects = [
            project
            for project in analysis.projects
            if project.project_id
        ]

        # Без проекта контекст проекта получать нельзя.
        if not resolved_projects:
            return

        for project in resolved_projects:
            project_id = project.project_id
            if not project_id:
                continue

            state = self.context_provider.get_state(project_id)
            history = self.context_provider.get_history(project_id)

            if state is None:
                # Новый проект: контекста ещё нет.
                continue

            self._resolve_requirement_replacements(
                message=message,
                analysis=analysis,
                project=project,
                state=state,
            )

            self._link_tasks_to_same_message_requirements(
                analysis=analysis,
                project=project,
            )

            self._resolve_short_reply_decisions(
                message=message,
                analysis=analysis,
                project=project,
                state=state,
                history=history,
            )

    def _resolve_requirement_replacements(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
        project: ProjectReference,
        state: Any,
    ) -> None:
        """
        Связывает каждую конструкцию "НОВОЕ вместо СТАРОГО"
        с соответствующим активным requirement в состоянии проекта.

        Поддерживает несколько замен в одном сообщении. Если для
        конкретной замены найдено не ровно одно старое требование,
        MIRA не угадывает и оставляет requires_human=True.
        """

        replacement_pattern = re.compile(
            r"(?P<prefix>[^.!?;]{0,80}?)"
            r"(?P<new>\d+(?:[.,]\d+)?)\s*(?P<new_unit>мм|см|м)?"
            r"\s+вместо\s+"
            r"(?P<old>\d+(?:[.,]\d+)?)\s*(?P<old_unit>мм|см|м)?",
            re.IGNORECASE,
        )
        replacements = list(
            replacement_pattern.finditer(message.text or "")
        )

        if not replacements:
            return

        unresolved = False

        for replacement in replacements:
            new_value = self._normalize_number_text(replacement.group("new"))
            old_value = self._normalize_number_text(replacement.group("old"))
            new_unit = replacement.group("new_unit") or replacement.group("old_unit")
            old_unit = replacement.group("old_unit") or replacement.group("new_unit")

            prefix = replacement.group("prefix").strip()
            prefix = re.sub(
                r"\b(теперь|делаем|сделать|будет|должен быть|должна быть)\b",
                " ",
                prefix,
                flags=re.IGNORECASE,
            )
            prefix_words = re.findall(r"[а-яёa-z0-9_-]+", prefix.lower())
            entity = prefix_words[-1] if prefix_words else None

            # Находим новое requirement именно для этой replacement-конструкции.
            new_candidates = []
            for item in analysis.requirements:
                if not self._same_project(item.project, project):
                    continue
                if not item.requirement_id or "_replacement" not in item.requirement_id:
                    continue
                if item.supersedes_requirement_id is not None:
                    continue
                if self._normalize_number_text(item.value) != new_value:
                    continue
                if new_unit and item.unit and item.unit.lower() != new_unit.lower():
                    continue
                if entity and item.entity and item.entity.lower() != entity.lower():
                    continue
                new_candidates.append(item)

            old_candidates = []
            for old_requirement in state.active_requirements:
                if not old_requirement.requirement_id:
                    continue
                if self._normalize_number_text(old_requirement.value) != old_value:
                    continue
                if (
                    old_unit
                    and old_requirement.unit
                    and old_requirement.unit.lower() != old_unit.lower()
                ):
                    continue
                # Если entity известна с обеих сторон, используем её как
                # дополнительный фильтр, чтобы не спутать одинаковые размеры.
                if (
                    entity
                    and old_requirement.entity
                    and old_requirement.entity.lower() != entity.lower()
                ):
                    continue
                old_candidates.append(old_requirement)

            if len(new_candidates) == 1 and len(old_candidates) == 1:
                new_requirement = new_candidates[0]
                old_requirement = old_candidates[0]

                new_requirement.supersedes_requirement_id = (
                    old_requirement.requirement_id
                )

                if not new_requirement.entity and old_requirement.entity:
                    new_requirement.entity = old_requirement.entity

                new_requirement.requires_human_confirmation = False
                new_requirement.confidence = max(
                    new_requirement.confidence,
                    0.95,
                )
            else:
                unresolved = True

        if unresolved:
            analysis.requires_human = True
            if not analysis.human_reason:
                analysis.human_reason = (
                    "Не удалось однозначно определить, "
                    "какое прежнее требование заменяется новым."
                )

    def _link_tasks_to_same_message_requirements(
        self,
        analysis: MessageAnalysis,
        project: ProjectReference,
    ) -> None:
        """
        Связывает задачу с requirement из того же сообщения,
        только если связь однозначна.

        Например:
            requirement: шкаф 250 мм
            task: поменять шкаф и прислать пререндер

        Пока это консервативное правило:
        если в проекте сообщения ровно одно новое requirement,
        оно добавляется в related_requirement_ids задачи.

        При нескольких requirements связь позже должен определять LLM.
        """

        requirements = [
            item
            for item in analysis.requirements
            if self._same_project(item.project, project)
            and item.requirement_id
        ]

        tasks = [
            item
            for item in analysis.tasks
            if self._same_project(item.project, project)
        ]

        if len(requirements) != 1:
            return

        requirement_id = requirements[0].requirement_id
        if not requirement_id:
            return

        for task in tasks:
            if requirement_id not in task.related_requirement_ids:
                task.related_requirement_ids.append(requirement_id)

    def _resolve_short_reply_decisions(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
        project: ProjectReference,
        state: Any,
        history: List[Any],
    ) -> None:
        """
        Интерпретирует короткое сообщение пользователя
        только через reply/context.

        Пример:
            архитектор: "Шкаф теперь 250 вместо 350."
            пользователь: "Поняла"

        Если reply_to_message_id указывает на сообщение,
        которое создало активный requirement, Decision связывается
        именно с этим requirement_id.

        Это позволяет ProjectStateEngine перевести requirement
        в ACCEPTED.
        """

        if message.speaker != Speaker.USER:
            return

        normalized = self._normalize_short_reply_text(message.text)

        short_accepts = {
            "поняла",
            "понял",
            "да",
            "хорошо",
            "ок",
            "окей",
            "ясно",
            "сделаю",
            "принято",
        }

        if normalized not in short_accepts:
            return

        if not analysis.decisions:
            return

        reply_to = None
        if message.context is not None:
            reply_to = message.context.reply_to_message_id

        if not reply_to:
            return

        # Собираем requirement IDs, которые были созданы/изменены
        # сообщением, на которое ответил пользователь.
        candidate_ids = set()

        for history_item in history:
            if history_item.source_message_id != reply_to:
                continue

            if history_item.change_type not in {
                "requirement_upserted",
                "requirement_replaced",
            }:
                continue

            if history_item.entity_id:
                candidate_ids.add(history_item.entity_id)

        # requirement_replaced в history указывает также на старый ID.
        # Нам нужны только те, которые всё ещё актуальны в state.
        active_ids = {
            req.requirement_id
            for req in state.active_requirements
            if req.requirement_id
        }

        candidate_ids &= active_ids

        if len(candidate_ids) == 1:
            requirement_id = next(iter(candidate_ids))

            decision = analysis.decisions[0]
            decision.related_requirement_ids = [
                requirement_id
            ]
            decision.status = DecisionStatus.CONFIRMED
            decision.confirmed_by_user = True
            decision.inferred_from_context = True
            decision.confidence = max(
                decision.confidence,
                0.98,
            )

            return

        # Reply есть, но невозможно однозначно понять,
        # что именно пользователь подтвердил.
        analysis.requires_human = True

        if not analysis.human_reason:
            analysis.human_reason = (
                "Ответ связан с предыдущим сообщением, "
                "но в нём несколько возможных требований."
            )

    @staticmethod
    def _normalize_short_reply_text(
        value: Optional[str],
    ) -> str:
        """
        Нормализует короткие текстовые подтверждения.

        Примеры:
            "Да."      -> "да"
            "Ок!"      -> "ок"
            "Хорошо 😊" -> "хорошо"

        Внутреннюю пунктуацию не удаляем:
            "Да, конечно" остаётся "да, конечно"
        и не превращается автоматически в короткое подтверждение.

        Standalone emoji / Telegram reactions обрабатываются отдельно
        на уровне acknowledgement/reaction context и сюда не подмешиваются.
        """
        raw = (value or "").strip().lower()
        raw = re.sub(r"\s+", " ", raw)

        # Снимаем только внешнюю пунктуацию/emoji.
        # Кириллица и латиница являются Unicode word characters,
        # поэтому "Да." безопасно превращается в "да".
        raw = re.sub(r"^\W+|\W+$", "", raw, flags=re.UNICODE)

        return raw

    @staticmethod
    def _normalize_number_text(
        value: Optional[str],
    ) -> Optional[str]:
        """
        Нормализует числовое значение для сравнения:
            "350" -> "350"
            "350,0" -> "350"
            "350.00" -> "350"
        """
        if value is None:
            return None

        raw = str(value).strip().replace(",", ".")

        try:
            number = float(raw)
        except ValueError:
            return raw

        if number.is_integer():
            return str(int(number))

        return str(number)

    @staticmethod
    def _same_project(
        left: Optional[ProjectReference],
        right: Optional[ProjectReference],
    ) -> bool:
        if left is None or right is None:
            return False

        return bool(
            left.project_id
            and right.project_id
            and left.project_id == right.project_id
        )

    # -----------------------------------------------------------------
    # SAFETY / CONTEXT
    # -----------------------------------------------------------------

    def _apply_context_safety(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> None:
        """
        Не даём короткому ответу пользователя автоматически
        становиться подтверждением, если reply/context не разрешён.
        """

        text = self._normalize_short_reply_text(message.text)

        short_replies = {
            "поняла",
            "понял",
            "да",
            "хорошо",
            "ок",
            "окей",
            "ясно",
            "сделаю",
            "принято",
        }

        if (
            message.speaker == Speaker.USER
            and text in short_replies
            and self.config.short_reply_requires_context
            and not self._has_resolved_context(message.context)
        ):
            analysis.requires_human = True

            if not analysis.human_reason:
                analysis.human_reason = (
                    "Короткий ответ нельзя интерпретировать "
                    "без связанного сообщения или контекста."
                )

    def _apply_deadline_safety(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> None:
        """
        Срок, заданный клиентом/архитектором/менеджером,
        не считается автоматически подтверждённым дизайнером.
        """

        if message.speaker == Speaker.USER:
            return

        has_external_deadline = any(
            task.deadline_iso
            and not task.deadline_confirmed_by_user
            for task in analysis.tasks
        )

        if (
            has_external_deadline
            and self.config.external_deadline_requires_confirmation
        ):
            analysis.requires_human = True

            if not analysis.human_reason:
                analysis.human_reason = (
                    "В сообщении указан новый срок, "
                    "который пользователь ещё не подтвердил."
                )

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _single_project_or_none(
        projects: List[ProjectReference],
    ) -> Optional[ProjectReference]:
        """
        Возвращает проект только если он один и определён уверенно.

        Если проектов несколько — нельзя автоматически присваивать
        одну и ту же задачу всем проектам.
        """
        resolved = [
            project
            for project in projects
            if project.project_id
        ]

        if len(resolved) == 1:
            return resolved[0]

        return None

    @staticmethod
    def _has_resolved_project(
        projects: List[ProjectReference],
    ) -> bool:
        return any(
            project.project_id
            for project in projects
        )

    @staticmethod
    def _has_resolved_context(
        context: Optional[MessageContext],
    ) -> bool:
        if context is None:
            return False

        return bool(
            context.context_resolved
            or context.reply_to_message_id
        )

    @staticmethod
    def _unique_intents(
        intents: List[IntentItem],
    ) -> List[IntentItem]:
        """
        Удаляет повторяющиеся intent одного типа.
        """
        result: List[IntentItem] = []
        seen = set()

        for item in intents:
            key = item.intent
            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    @staticmethod
    def _collect_affected_project_ids(
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ) -> List[str]:
        """
        Собирает все явно определённые project_id.

        Никаких догадок:
        project_id=None просто игнорируется.
        """

        ids = set()

        def add_project(project: Optional[ProjectReference]) -> None:
            if project and project.project_id:
                ids.add(project.project_id)

        for project in message.projects:
            add_project(project)

        for project in analysis.projects:
            add_project(project)

        for item in analysis.intents:
            add_project(item.project)

        for item in analysis.tasks:
            add_project(item.project)

        for item in analysis.requirements:
            add_project(item.project)

        for item in analysis.decisions:
            add_project(item.project)

        for item in analysis.agreements:
            add_project(item.project)

        for item in analysis.financial_issues:
            add_project(item.project)

        for item in analysis.events:
            add_project(item.project)

        return sorted(ids)


# ---------------------------------------------------------------------
# БЫСТРАЯ ДЕМОНСТРАЦИЯ
# ---------------------------------------------------------------------

if __name__ == "__main__":

    project = ProjectReference(
        project_id="villa_2",
        project_name="Villa 2",
        confidence=0.99,
    )

    message = MessageEnvelope(
        message_id="demo_001",
        dialog_id="dialog_001",
        contact_id="architect_001",
        speaker=Speaker.CONTACT,
        contact_role=ContactRole.ARCHITECT,
        timestamp=datetime(2026, 8, 30, 12, 0),
        text=(
            "По второй вилле шкаф теперь 250 вместо 350. "
            "Пришли новый пререндер завтра до 18:00."
        ),
        projects=[project],
    )

    engine = AnalysisEngine()
    result = engine.analyze(message)

    print("Проекты:", result.affected_project_ids)

    print("Намерения:")
    for item in result.analysis.intents:
        print(" -", item.intent.value)

    print("Требования:")
    for item in result.analysis.requirements:
        print(" -", item.description)

    print("Задачи:")
    for item in result.analysis.tasks:
        print(
            " -",
            item.description,
            "| deadline:",
            item.deadline_iso,
        )

    print("Требуется участие пользователя:")
    print(result.analysis.requires_human)

    if result.analysis.human_reason:
        print("Причина:")
        print(result.analysis.human_reason)
