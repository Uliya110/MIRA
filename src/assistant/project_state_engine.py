"""
MIRA — Project State Engine
===========================

Этот модуль обновляет текущее состояние проекта на основе уже
структурированного результата анализа сообщения (`AnalysisResult`).

Важно:
- этот файл НЕ занимается пониманием сырого текста;
- он НЕ вызывает RuBERT/LLM;
- он НЕ анализирует PDF/изображения/ссылки;
- он получает уже готовые сущности из `analysis_schema_v2.py`
  и применяет их к состоянию проекта.

Проще говоря:

    сообщение
        ↓
    Analysis Engine
        ↓
    AnalysisResult
        ↓
    ProjectStateEngine
        ↓
    актуальное состояние проекта

Главная задача движка:
не просто хранить историю сообщений, а понимать, что СЕЙЧАС активно
для проекта:
- какие требования действуют;
- какие требования заменены;
- какие задачи открыты / приостановлены / завершены;
- какие решения подтверждены;
- какие договорённости активны;
- какие вопросы пока не закрыты.

Все пользовательские тексты MIRA должны оставаться на русском языке.
Внутренние имена классов и полей могут быть английскими.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from analysis_schema_v2 import (
    Agreement,
    AnalysisResult,
    ReminderItem,
    ReminderKind,
    ReminderStatus,
    Decision,
    DecisionStatus,
    MessageEventType,
    ProjectReference,
    ProjectState,
    Requirement,
    RequirementStatus,
    TaskItem,
    TaskStatus,
)


# ---------------------------------------------------------------------
# ДОПОЛНИТЕЛЬНЫЕ ВНУТРЕННИЕ СТРУКТУРЫ
# ---------------------------------------------------------------------

@dataclass
class StateChange:
    """
    Одна зафиксированная операция изменения состояния проекта.

    Это полезно для отладки и будущего аудита.
    Например:
        requirement_replaced
        task_paused
        agreement_added
    """
    project_id: str
    change_type: str
    entity_id: Optional[str] = None
    description: Optional[str] = None
    source_message_id: Optional[str] = None


@dataclass
class ProjectHistoryEntry:
    """
    Неизменяемая запись истории проекта.

    History не заменяет текущее состояние. Она объясняет,
    как проект пришёл к текущему состоянию.
    """
    project_id: str
    timestamp: datetime
    change_type: str

    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    description: Optional[str] = None

    source_message_id: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None


@dataclass
class ProjectStateUpdate:
    """
    Результат применения одного AnalysisResult.

    states:
        обновлённые состояния затронутых проектов.

    changes:
        список изменений, которые были внесены движком.

    warnings:
        ситуации, которые не являются ошибкой программы,
        но требуют внимания.
    """
    states: Dict[str, ProjectState] = field(default_factory=dict)
    changes: List[StateChange] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# ОСНОВНОЙ ДВИЖОК
# ---------------------------------------------------------------------

class ProjectStateEngine:
    """
    Управляет текущими состояниями проектов.

    В MVP состояния хранятся в памяти Python в словаре self._states.

    Позже этот слой можно заменить или расширить:
    - PostgreSQL;
    - SQLite;
    - Redis;
    - отдельным Repository/Storage слоем.

    При этом сама логика обновления состояния останется здесь.
    """

    def __init__(self) -> None:
        # Ключ: project_id
        # Значение: актуальный ProjectState
        self._states: Dict[str, ProjectState] = {}

        # Полная последовательность изменений по каждому проекту.
        self._history: Dict[str, List[ProjectHistoryEntry]] = {}

        # Активные напоминания по каждому проекту.
        self._reminders: Dict[str, List[ReminderItem]] = {}

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def get_state(self, project_id: str) -> Optional[ProjectState]:
        """
        Возвращает копию состояния проекта.

        Почему копию:
        чтобы внешний код случайно не изменил внутреннее состояние
        движка напрямую.
        """
        state = self._states.get(project_id)

        if state is None:
            return None

        return deepcopy(state)

    def get_all_states(self) -> Dict[str, ProjectState]:
        """
        Возвращает копии всех известных состояний проектов.
        """
        return deepcopy(self._states)

    def get_history(
        self,
        project_id: str,
    ) -> List[ProjectHistoryEntry]:
        """
        Возвращает историю проекта в хронологическом порядке.
        """
        return deepcopy(self._history.get(project_id, []))

    def get_reminders(
        self,
        project_id: str,
        pending_only: bool = True,
    ) -> List[ReminderItem]:
        """
        Возвращает напоминания проекта.

        По умолчанию — только ожидающие отправки.
        """
        reminders = deepcopy(self._reminders.get(project_id, []))

        if pending_only:
            reminders = [
                item
                for item in reminders
                if item.status == ReminderStatus.PENDING
            ]

        return reminders

    def apply(self, result: AnalysisResult) -> ProjectStateUpdate:
        """
        Главный метод.

        Применяет один AnalysisResult к состояниям всех затронутых проектов.

        Важное правило:
        если проект определить нельзя, движок НЕ угадывает project_id.
        Такая ситуация возвращается как warning и должна быть показана
        пользователю через вышестоящий слой.
        """
        update = ProjectStateUpdate()

        project_refs = self._collect_project_references(result)

        if not project_refs:
            update.warnings.append(
                "Не удалось определить проект для обновления состояния."
            )
            return update

        for project_ref in project_refs:
            if not project_ref.project_id:
                update.warnings.append(
                    "Обнаружена сущность с неопределённым проектом. "
                    "Автоматическое обновление состояния пропущено."
                )
                continue

            state = self._get_or_create_state(project_ref)

            # Контакт автора сообщения добавляем в участников проекта.
            self._add_contact_if_missing(
                state=state,
                contact_id=result.message.contact_id,
            )

            # Применяем структурированные сущности.
            self._apply_requirements(result, state, update)
            self._apply_tasks(result, state, update)
            self._apply_decisions(result, state, update)
            self._apply_agreements(result, state, update)
            self._apply_events(result, state, update)

            # Сроки задач, договорённостей и будущих событий
            # превращаем в напоминания.
            self._apply_reminders(result, state, update)

            # Все изменения этого сообщения сохраняем в историю.
            self._append_history_from_changes(
                state=state,
                result=result,
                changes=[
                    c for c in update.changes
                    if c.project_id == state.project_id
                ],
            )

            # Сохраняем время последнего изменения.
            state.last_updated_at = (
                result.message.timestamp
                if result.message.timestamp is not None
                else datetime.now()
            )

            self._states[state.project_id] = state
            update.states[state.project_id] = deepcopy(state)

        return update

    # -----------------------------------------------------------------
    # PROJECT RESOLUTION
    # -----------------------------------------------------------------

    def _collect_project_references(
        self,
        result: AnalysisResult,
    ) -> List[ProjectReference]:
        """
        Собирает все project references из AnalysisResult.

        Почему нельзя опираться только на message.projects:
        одна задача внутри сообщения может относиться к одному проекту,
        другая — к другому.

        Поэтому собираем проекты из:
        - сообщения;
        - анализа;
        - tasks;
        - requirements;
        - decisions;
        - agreements;
        - financial issues;
        - events.

        Затем удаляем дубликаты по project_id.
        """
        refs: List[ProjectReference] = []

        refs.extend(result.message.projects)
        refs.extend(result.analysis.projects)

        for item in result.analysis.tasks:
            if item.project is not None:
                refs.append(item.project)

        for item in result.analysis.requirements:
            if item.project is not None:
                refs.append(item.project)

        for item in result.analysis.decisions:
            if item.project is not None:
                refs.append(item.project)

        for item in result.analysis.agreements:
            if item.project is not None:
                refs.append(item.project)

        for item in result.analysis.financial_issues:
            if item.project is not None:
                refs.append(item.project)

        for item in result.analysis.events:
            if item.project is not None:
                refs.append(item.project)

        unique: Dict[str, ProjectReference] = {}

        # Ссылки без project_id не склеиваем в один "unknown" проект.
        # Они отдельно обрабатываются как неопределённые.
        unknown_refs: List[ProjectReference] = []

        for ref in refs:
            if ref.project_id:
                # Если проект уже встречался, сохраняем вариант
                # с большей уверенностью.
                existing = unique.get(ref.project_id)

                if (
                    existing is None
                    or ref.confidence > existing.confidence
                ):
                    unique[ref.project_id] = ref
            else:
                unknown_refs.append(ref)

        return list(unique.values()) + unknown_refs

    def _get_or_create_state(
        self,
        project_ref: ProjectReference,
    ) -> ProjectState:
        """
        Возвращает существующее состояние проекта
        или создаёт новое.
        """
        assert project_ref.project_id is not None

        existing = self._states.get(project_ref.project_id)

        if existing is not None:
            return existing

        project_name = (
            project_ref.project_name
            or project_ref.project_id
        )

        state = ProjectState(
            project_id=project_ref.project_id,
            project_name=project_name,
        )

        self._states[state.project_id] = state
        return state

    @staticmethod
    def _add_contact_if_missing(
        state: ProjectState,
        contact_id: str,
    ) -> None:
        """
        Добавляет участника проекта без дублей.
        """
        if contact_id not in state.contact_ids:
            state.contact_ids.append(contact_id)

    # -----------------------------------------------------------------
    # REQUIREMENTS
    # -----------------------------------------------------------------

    def _apply_requirements(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Применяет требования сообщения к конкретному проекту.

        Основные сценарии:
        1. Новое требование -> добавляем в active_requirements.
        2. Новое требование заменяет старое ->
           старое переводим в REPLACED, новое становится активным.
           Связанные со старым требованием задачи помечаем UNCLEAR,
           потому что их содержание может быть устаревшим.
        3. Требование CANCELLED/COMPLETED ->
           больше не считаем его активным.
        """
        project_requirements = [
            req
            for req in result.analysis.requirements
            if self._belongs_to_state(req.project, state)
        ]

        for requirement in project_requirements:

            # ---------------------------------------------------------
            # Если новое требование явно заменяет старое.
            # ---------------------------------------------------------
            if requirement.supersedes_requirement_id:
                old = self._find_requirement(
                    state=state,
                    requirement_id=requirement.supersedes_requirement_id,
                )

                if old is not None:
                    old.status = RequirementStatus.REPLACED

                    self._remove_requirement_from_active(
                        state=state,
                        requirement_id=old.requirement_id,
                    )

                    update.changes.append(
                        StateChange(
                            project_id=state.project_id,
                            change_type="requirement_replaced",
                            entity_id=old.requirement_id,
                            description=(
                                f"Требование заменено: {old.description}"
                            ),
                            source_message_id=requirement.source_message_id,
                        )
                    )

                    # -------------------------------------------------
                    # ВАЖНО:
                    # старая задача не должна продолжать выглядеть
                    # актуальной, если она была построена на заменённом
                    # требовании.
                    #
                    # Мы НЕ переписываем текст задачи автоматически:
                    # ProjectStateEngine не должен придумывать новое
                    # содержание. Он только помечает задачу как UNCLEAR,
                    # а Analysis Engine должен создать/обновить задачу
                    # уже по новому требованию.
                    # -------------------------------------------------
                    self._invalidate_tasks_for_requirement(
                        state=state,
                        requirement_id=old.requirement_id,
                        update=update,
                        source_message_id=requirement.source_message_id,
                    )
                else:
                    update.warnings.append(
                        f"Не найдено заменяемое требование "
                        f"{requirement.supersedes_requirement_id} "
                        f"для проекта {state.project_name}."
                    )

            # ---------------------------------------------------------
            # Неактивное требование не кладём в active_requirements.
            # ---------------------------------------------------------
            if requirement.status in {
                RequirementStatus.CANCELLED,
                RequirementStatus.COMPLETED,
                RequirementStatus.REPLACED,
            }:
                self._remove_requirement_from_active(
                    state=state,
                    requirement_id=requirement.requirement_id,
                )
                continue

            # NEW / ACTIVE / ACCEPTED / UNCLEAR
            self._upsert_requirement(
                state=state,
                requirement=requirement,
            )

            update.changes.append(
                StateChange(
                    project_id=state.project_id,
                    change_type="requirement_upserted",
                    entity_id=requirement.requirement_id,
                    description=requirement.description,
                    source_message_id=requirement.source_message_id,
                )
            )

    # -----------------------------------------------------------------
    # TASKS
    # -----------------------------------------------------------------

    def _apply_tasks(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Обновляет список текущих задач проекта.

        В open_tasks оставляем:
        - NEW
        - IN_PROGRESS
        - PAUSED
        - WAITING_FOR_CONTACT
        - WAITING_FOR_USER
        - UNCLEAR

        Не оставляем:
        - COMPLETED
        - CANCELLED
        """
        project_tasks = [
            task
            for task in result.analysis.tasks
            if self._belongs_to_state(task.project, state)
        ]

        for task in project_tasks:

            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
            }:
                self._remove_task(
                    state=state,
                    task_id=task.task_id,
                )

                update.changes.append(
                    StateChange(
                        project_id=state.project_id,
                        change_type=f"task_{task.status.value}",
                        entity_id=task.task_id,
                        description=task.description,
                        source_message_id=task.source_message_id,
                    )
                )
                continue

            self._upsert_task(
                state=state,
                task=task,
            )

            update.changes.append(
                StateChange(
                    project_id=state.project_id,
                    change_type=f"task_{task.status.value}",
                    entity_id=task.task_id,
                    description=task.description,
                    source_message_id=task.source_message_id,
                )
            )

    # -----------------------------------------------------------------
    # DECISIONS
    # -----------------------------------------------------------------

    def _apply_decisions(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Решения сами по себе пока не хранятся отдельным списком
        внутри ProjectState v2.2.

        Но подтверждённое решение может менять статус требования.

        Например:
            клиент: "стекло 2400"
            пользователь: "поняла"

        Analysis Engine создаёт Decision:
            confirmed_by_user=True
            related_requirement_ids=["req_glass"]

        Тогда движок переводит связанные требования в ACCEPTED.
        """
        project_decisions = [
            decision
            for decision in result.analysis.decisions
            if self._belongs_to_state(decision.project, state)
        ]

        for decision in project_decisions:

            if (
                decision.status == DecisionStatus.CONFIRMED
                and decision.confirmed_by_user
            ):
                for requirement_id in decision.related_requirement_ids:
                    requirement = self._find_requirement(
                        state=state,
                        requirement_id=requirement_id,
                    )

                    if requirement is None:
                        # Требование может существовать в БД/истории,
                        # но не быть загружено в текущий in-memory state.
                        update.warnings.append(
                            f"Решение ссылается на требование "
                            f"{requirement_id}, которого нет в текущем "
                            f"состоянии проекта {state.project_name}."
                        )
                        continue

                    requirement.status = RequirementStatus.ACCEPTED

                    update.changes.append(
                        StateChange(
                            project_id=state.project_id,
                            change_type="requirement_accepted",
                            entity_id=requirement.requirement_id,
                            description=requirement.description,
                            source_message_id=decision.source_message_id,
                        )
                    )

    # -----------------------------------------------------------------
    # AGREEMENTS
    # -----------------------------------------------------------------

    def _apply_agreements(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Активные договорённости храним в active_agreements.

        Если agreement.active=False, удаляем его из активных.
        Историю в production-версии нужно сохранять отдельно.
        """
        project_agreements = [
            agreement
            for agreement in result.analysis.agreements
            if self._belongs_to_state(agreement.project, state)
        ]

        for agreement in project_agreements:

            if not agreement.active:
                self._remove_agreement(
                    state=state,
                    agreement_id=agreement.agreement_id,
                )

                update.changes.append(
                    StateChange(
                        project_id=state.project_id,
                        change_type="agreement_deactivated",
                        entity_id=agreement.agreement_id,
                        description=agreement.description,
                    )
                )
                continue

            self._upsert_agreement(
                state=state,
                agreement=agreement,
            )

            update.changes.append(
                StateChange(
                    project_id=state.project_id,
                    change_type="agreement_upserted",
                    entity_id=agreement.agreement_id,
                    description=agreement.description,
                )
            )

    # -----------------------------------------------------------------
    # EVENTS
    # -----------------------------------------------------------------

    def _apply_events(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Некоторые события напрямую меняют состояние задач.

        Пример:
        TASK_PAUSED -> соответствующая задача должна стать PAUSED.

        Пока event не содержит related_task_ids, поэтому автоматическое
        сопоставление ограничено.

        Это намеренно:
        движок не должен угадывать связь между событием и задачей,
        если Analysis Engine её явно не зафиксировал.
        """
        project_events = [
            event
            for event in result.analysis.events
            if self._belongs_to_state(event.project, state)
        ]

        for event in project_events:
            update.changes.append(
                StateChange(
                    project_id=state.project_id,
                    change_type=f"event_{event.event_type.value}",
                    entity_id=event.event_id,
                    description=event.description,
                    source_message_id=event.source_message_id,
                )
            )

            # На этом этапе фиксируем событие в журнал изменений.
            # Более сложные переходы будут добавляться,
            # когда в schema появятся related_task_ids / related_result_ids.

    # -----------------------------------------------------------------
    # HISTORY
    # -----------------------------------------------------------------

    def _append_history_from_changes(
        self,
        state: ProjectState,
        result: AnalysisResult,
        changes: List[StateChange],
    ) -> None:
        """
        Сохраняет журнал изменений как отдельную историю проекта.

        Важно:
        active_requirements/open_tasks показывают "что актуально сейчас",
        а history — "что происходило раньше".
        """
        timestamp = result.message.timestamp or datetime.now()

        history = self._history.setdefault(state.project_id, [])

        existing_keys = {
            (
                item.source_message_id,
                item.change_type,
                item.entity_id,
                item.description,
            )
            for item in history
        }

        for change in changes:
            key = (
                change.source_message_id,
                change.change_type,
                change.entity_id,
                change.description,
            )

            # Защита от случайной повторной обработки одного сообщения.
            if key in existing_keys:
                continue

            history.append(
                ProjectHistoryEntry(
                    project_id=state.project_id,
                    timestamp=timestamp,
                    change_type=change.change_type,
                    entity_id=change.entity_id,
                    description=change.description,
                    source_message_id=change.source_message_id,
                )
            )
            existing_keys.add(key)

    # -----------------------------------------------------------------
    # REMINDERS
    # -----------------------------------------------------------------

    def _apply_reminders(
        self,
        result: AnalysisResult,
        state: ProjectState,
        update: ProjectStateUpdate,
    ) -> None:
        """
        Создаёт напоминания, если анализ сообщения содержит
        нормализованную дату/время.

        Источники:
        - TaskItem.deadline_iso
        - Agreement.deadline_iso
        - MessageEvent.scheduled_for_iso

        Сам ProjectStateEngine НЕ запускает таймер и НЕ отправляет
        уведомления. Он только создаёт корректные ReminderItem.

        Фактическую доставку позже будет выполнять отдельный
        ReminderService / scheduler.
        """
        reminders = self._reminders.setdefault(state.project_id, [])

        # ------------------------------
        # СРОКИ ЗАДАЧ
        # ------------------------------
        for task in result.analysis.tasks:
            if not self._belongs_to_state(task.project, state):
                continue

            if not task.deadline_iso:
                continue

            reminder = self._build_reminder(
                project_id=state.project_id,
                kind=ReminderKind.TASK_DEADLINE,
                due_at_iso=task.deadline_iso,
                source_message_id=task.source_message_id,
                source_entity_id=task.task_id,
                user_message_ru=(
                    f"Напоминание по проекту «{state.project_name}»: "
                    f"срок задачи «{task.description}» — "
                    f"{task.deadline_text or task.deadline_iso}."
                ),
            )
            self._upsert_reminder(reminders, reminder)

        # ------------------------------
        # ДОГОВОРЁННОСТИ
        # ------------------------------
        for agreement in result.analysis.agreements:
            if not self._belongs_to_state(agreement.project, state):
                continue

            if not agreement.active:
                self._cancel_reminders_for_entity(
                    reminders,
                    agreement.agreement_id,
                )
                continue

            if not agreement.deadline_iso:
                continue

            reminder = self._build_reminder(
                project_id=state.project_id,
                kind=ReminderKind.AGREEMENT_DEADLINE,
                due_at_iso=agreement.deadline_iso,
                source_message_id=(
                    agreement.source_message_ids[-1]
                    if agreement.source_message_ids
                    else result.message.message_id
                ),
                source_entity_id=agreement.agreement_id,
                user_message_ru=(
                    f"Напоминание по проекту «{state.project_name}»: "
                    f"договорённость «{agreement.description}» "
                    f"назначена на "
                    f"{agreement.deadline_text or agreement.deadline_iso}."
                ),
            )
            self._upsert_reminder(reminders, reminder)

        # ------------------------------
        # БУДУЩИЕ СОБЫТИЯ
        # ------------------------------
        for event in result.analysis.events:
            if not self._belongs_to_state(event.project, state):
                continue

            if not event.scheduled_for_iso:
                continue

            reminder = self._build_reminder(
                project_id=state.project_id,
                kind=ReminderKind.SCHEDULED_EVENT,
                due_at_iso=event.scheduled_for_iso,
                source_message_id=event.source_message_id,
                source_entity_id=event.event_id,
                user_message_ru=(
                    f"Напоминание по проекту «{state.project_name}»: "
                    f"{event.description or 'запланировано событие'} — "
                    f"{event.scheduled_for_text or event.scheduled_for_iso}."
                ),
            )
            self._upsert_reminder(reminders, reminder)

    @staticmethod
    def _build_reminder(
        project_id: str,
        kind: ReminderKind,
        due_at_iso: str,
        source_message_id: Optional[str],
        source_entity_id: Optional[str],
        user_message_ru: str,
        notify_before_minutes: int = 60,
    ) -> ReminderItem:
        """
        Строит напоминание.

        Если due_at_iso содержит корректный ISO datetime,
        trigger_at ставится за notify_before_minutes до срока.

        Если это только дата без времени, используем 09:00 этого дня.
        """
        due = datetime.fromisoformat(due_at_iso)

        # Если указана только дата, fromisoformat даст 00:00.
        # Для пользовательского напоминания это слишком рано,
        # поэтому переносим технический due на 09:00 этого дня.
        if (
            due.hour == 0
            and due.minute == 0
            and "T" not in due_at_iso
            and " " not in due_at_iso
        ):
            due = due.replace(hour=9)

        trigger = due - timedelta(minutes=notify_before_minutes)

        raw_id = (
            f"{project_id}|{kind.value}|"
            f"{source_entity_id or source_message_id}|{due.isoformat()}"
        )

        # Стабильный читаемый ID без внешних зависимостей.
        reminder_id = "rem_" + str(abs(hash(raw_id)))

        return ReminderItem(
            reminder_id=reminder_id,
            project_id=project_id,
            kind=kind,
            trigger_at_iso=trigger.isoformat(),
            due_at_iso=due.isoformat(),
            source_message_id=source_message_id,
            source_entity_id=source_entity_id,
            user_message_ru=user_message_ru,
            status=ReminderStatus.PENDING,
            notify_before_minutes=notify_before_minutes,
        )

    @staticmethod
    def _upsert_reminder(
        reminders: List[ReminderItem],
        reminder: ReminderItem,
    ) -> None:
        """
        Не создаёт дубликаты одного и того же напоминания.
        """
        reminders[:] = [
            item
            for item in reminders
            if item.reminder_id != reminder.reminder_id
        ]
        reminders.append(reminder)

    @staticmethod
    def _cancel_reminders_for_entity(
        reminders: List[ReminderItem],
        entity_id: Optional[str],
    ) -> None:
        """
        Если договорённость отменена, связанные напоминания
        не должны продолжать ожидать отправки.
        """
        if not entity_id:
            return

        for reminder in reminders:
            if reminder.source_entity_id == entity_id:
                reminder.status = ReminderStatus.CANCELLED

    # -----------------------------------------------------------------
    # HELPERS: REQUIREMENTS
    # -----------------------------------------------------------------

    @staticmethod
    def _find_requirement(
        state: ProjectState,
        requirement_id: Optional[str],
    ) -> Optional[Requirement]:
        if not requirement_id:
            return None

        for requirement in state.active_requirements:
            if requirement.requirement_id == requirement_id:
                return requirement

        return None

    @staticmethod
    def _remove_requirement_from_active(
        state: ProjectState,
        requirement_id: Optional[str],
    ) -> None:
        if not requirement_id:
            return

        state.active_requirements = [
            item
            for item in state.active_requirements
            if item.requirement_id != requirement_id
        ]

    def _upsert_requirement(
        self,
        state: ProjectState,
        requirement: Requirement,
    ) -> None:
        """
        Обновляет существующее требование с тем же ID
        либо добавляет новое.
        """
        if requirement.requirement_id:
            self._remove_requirement_from_active(
                state=state,
                requirement_id=requirement.requirement_id,
            )

        state.active_requirements.append(deepcopy(requirement))

    def _invalidate_tasks_for_requirement(
        self,
        state: ProjectState,
        requirement_id: Optional[str],
        update: ProjectStateUpdate,
        source_message_id: Optional[str],
    ) -> None:
        """
        Помечает открытые задачи как UNCLEAR, если они были связаны
        с требованием, которое больше не является актуальным.

        Почему не CANCELLED:
        замена требования не всегда означает отмену всей задачи.

        Пример:
            было: "шкаф 350 мм"
            стало: "шкаф 250 мм"

        Сама задача "сделать шкаф" остаётся нужна, но её параметры
        изменились. Поэтому безопаснее поставить UNCLEAR и дождаться,
        когда Analysis Engine создаст или обновит корректную задачу.

        Это принцип human-in-the-loop:
        движок не переписывает смысл задачи сам.
        """
        if not requirement_id:
            return

        for task in state.open_tasks:
            if requirement_id not in task.related_requirement_ids:
                continue

            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
            }:
                continue

            task.status = TaskStatus.UNCLEAR
            task.requires_confirmation = True

            update.changes.append(
                StateChange(
                    project_id=state.project_id,
                    change_type="task_invalidated_by_requirement_change",
                    entity_id=task.task_id,
                    description=(
                        f"Задача требует обновления после изменения "
                        f"связанного требования: {task.description}"
                    ),
                    source_message_id=source_message_id,
                )
            )

    # -----------------------------------------------------------------
    # HELPERS: TASKS
    # -----------------------------------------------------------------

    @staticmethod
    def _remove_task(
        state: ProjectState,
        task_id: Optional[str],
    ) -> None:
        if not task_id:
            return

        state.open_tasks = [
            item
            for item in state.open_tasks
            if item.task_id != task_id
        ]

    def _upsert_task(
        self,
        state: ProjectState,
        task: TaskItem,
    ) -> None:
        if task.task_id:
            self._remove_task(
                state=state,
                task_id=task.task_id,
            )

        state.open_tasks.append(deepcopy(task))

    # -----------------------------------------------------------------
    # HELPERS: AGREEMENTS
    # -----------------------------------------------------------------

    @staticmethod
    def _remove_agreement(
        state: ProjectState,
        agreement_id: Optional[str],
    ) -> None:
        if not agreement_id:
            return

        state.active_agreements = [
            item
            for item in state.active_agreements
            if item.agreement_id != agreement_id
        ]

    def _upsert_agreement(
        self,
        state: ProjectState,
        agreement: Agreement,
    ) -> None:
        if agreement.agreement_id:
            self._remove_agreement(
                state=state,
                agreement_id=agreement.agreement_id,
            )

        state.active_agreements.append(deepcopy(agreement))

    # -----------------------------------------------------------------
    # HELPERS: COMMON
    # -----------------------------------------------------------------

    @staticmethod
    def _belongs_to_state(
        project: Optional[ProjectReference],
        state: ProjectState,
    ) -> bool:
        """
        Проверяет, относится ли сущность к текущему проекту.

        Если у сущности project=None, автоматически к проекту её
        НЕ приписываем — это безопаснее, чем ошибочно смешать контекст.
        """
        if project is None:
            return False

        return project.project_id == state.project_id


# ---------------------------------------------------------------------
# НЕБОЛЬШОЙ ПРИМЕР РАБОТЫ
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Пример:
    1. В проект приходит требование "шкаф 350 мм".
    2. Затем приходит новая правка "шкаф 250 мм вместо 350".
    3. Старое требование перестаёт быть активным.
    """

    from analysis_schema_v2 import (
        IntentItem,
        IntentType,
        MessageAnalysis,
        MessageEnvelope,
        Speaker,
        ContactRole,
    )

    engine = ProjectStateEngine()

    villa_2 = ProjectReference(
        project_id="villa_2",
        project_name="Villa 2",
        confidence=0.99,
    )

    # -------------------------------------------------------------
    # ШАГ 1. Исходное требование
    # -------------------------------------------------------------

    first_message = MessageEnvelope(
        message_id="msg_001",
        dialog_id="dialog_001",
        contact_id="architect_001",
        speaker=Speaker.CONTACT,
        contact_role=ContactRole.ARCHITECT,
        text="Шкаф делаем глубиной 350 мм.",
        projects=[villa_2],
    )

    first_requirement = Requirement(
        requirement_id="req_cabinet_depth_350",
        description="Глубина шкафа 350 мм",
        project=villa_2,
        source_message_id=first_message.message_id,
        status=RequirementStatus.ACTIVE,
        confidence=0.99,
        entity="шкаф",
        attribute="глубина",
        value="350",
        unit="мм",
    )

    first_analysis = MessageAnalysis(
        message_id=first_message.message_id,
        projects=[villa_2],
        intents=[
            IntentItem(
                intent=IntentType.TECHNICAL_QUESTION,
                confidence=0.95,
                project=villa_2,
            )
        ],
        requirements=[first_requirement],
        confidence=0.95,
    )

    first_result = AnalysisResult(
        message=first_message,
        analysis=first_analysis,
        affected_project_ids=["villa_2"],
    )

    update_1 = engine.apply(first_result)

    print("\nПосле первого сообщения:")
    state = update_1.states["villa_2"]

    for req in state.active_requirements:
        print("-", req.description, "|", req.status.value)

    # -------------------------------------------------------------
    # ШАГ 2. Новое требование заменяет старое
    # -------------------------------------------------------------

    second_message = MessageEnvelope(
        message_id="msg_002",
        dialog_id="dialog_001",
        contact_id="architect_001",
        speaker=Speaker.CONTACT,
        contact_role=ContactRole.ARCHITECT,
        text="Шкаф делаем 250 вместо 350.",
        projects=[villa_2],
    )

    second_requirement = Requirement(
        requirement_id="req_cabinet_depth_250",
        description="Глубина шкафа 250 мм",
        project=villa_2,
        source_message_id=second_message.message_id,
        status=RequirementStatus.ACTIVE,
        confidence=0.99,
        supersedes_requirement_id="req_cabinet_depth_350",
        entity="шкаф",
        attribute="глубина",
        value="250",
        unit="мм",
    )

    second_analysis = MessageAnalysis(
        message_id=second_message.message_id,
        projects=[villa_2],
        requirements=[second_requirement],
        confidence=0.99,
    )

    second_result = AnalysisResult(
        message=second_message,
        analysis=second_analysis,
        affected_project_ids=["villa_2"],
    )

    update_2 = engine.apply(second_result)

    print("\nПосле второй правки:")
    state = update_2.states["villa_2"]

    for req in state.active_requirements:
        print("-", req.description, "|", req.status.value)

    print("\nИзменения:")

    for change in update_2.changes:
        print("-", change.change_type, "|", change.description)
