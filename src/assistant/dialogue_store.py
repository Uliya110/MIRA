# -*- coding: utf-8 -*-
r"""
MIRA — Dialogue Store / Message Repository
==========================================

Рекомендуемое расположение:
    MIRA/src/assistant/dialogue_store.py

Назначение
----------
DialogueStore хранит ПОЛНУЮ двустороннюю историю сообщений.

Это отдельный слой от ProjectStateEngine:

    DialogueStore
        хранит исходные сообщения целиком
        ↓
    AnalysisEngine
        понимает новое сообщение
        ↓
    ProjectStateEngine
        хранит актуальные структурированные факты проекта

Ключевой принцип:
- история сообщений НЕ обрезается по мере роста диалога;
- при анализе модели передаётся не вся история, а выбранное контекстное окно;
- reply_to может быть старше обычного окна и всё равно должен быть включён;
- тексты, вложения, ссылки и metadata сохраняются без пересказа.

Текущая MVP-реализация:
- in-memory;
- без искусственного ограничения числа сообщений;
- данные живут до завершения процесса Python.

Для коммерческой версии этот же API можно перенести на SQLite/PostgreSQL
без изменения AnalysisEngine/Telegram-слоя.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .analysis_schema_v2 import MessageEnvelope


class DuplicateMessageError(ValueError):
    """Попытка молча перезаписать уже сохранённое сообщение."""


@dataclass(frozen=True)
class MessageKey:
    """
    message_id может повторяться в разных диалогах,
    поэтому ключ всегда составной.
    """

    dialog_id: str
    message_id: str


@dataclass
class DialogueContextBundle:
    """
    Набор сообщений, который можно передать следующему слою анализа.

    messages:
        выбранный контекст в порядке диалога.

    reply_target:
        исходное сообщение reply_to, если оно существует.

    recent_messages:
        последние сообщения перед текущим.

    Важно:
    это НЕ полный архив. Полный архив остаётся внутри DialogueStore.
    """

    dialog_id: str
    current_message_id: Optional[str]
    messages: List[MessageEnvelope]
    reply_target: Optional[MessageEnvelope]
    recent_messages: List[MessageEnvelope]


class DialogueStore:
    """
    Полное хранилище исходных сообщений MIRA.

    Никакого max_history здесь намеренно нет.
    Ограничение применяется только при ВЫБОРЕ контекста для модели.
    """

    def __init__(self) -> None:
        # Полная последовательность сообщений по dialog_id.
        self._dialogs: Dict[str, List[MessageEnvelope]] = {}

        # Быстрый доступ к конкретному сообщению.
        self._index: Dict[MessageKey, MessageEnvelope] = {}

        # Проект -> список ключей сообщений.
        self._project_index: Dict[str, List[MessageKey]] = {}

    # -----------------------------------------------------------------
    # WRITE API
    # -----------------------------------------------------------------

    def add_message(
        self,
        message: MessageEnvelope,
    ) -> None:
        """
        Сохраняет одно сообщение целиком.

        Молча перезаписывать существующее сообщение нельзя:
        это опасно для аудита договорённостей.
        """

        self._validate_message(message)

        key = MessageKey(
            dialog_id=message.dialog_id,
            message_id=message.message_id,
        )

        if key in self._index:
            raise DuplicateMessageError(
                "Сообщение уже существует: "
                f"dialog_id={message.dialog_id!r}, "
                f"message_id={message.message_id!r}"
            )

        stored = deepcopy(message)

        self._dialogs.setdefault(
            stored.dialog_id,
            [],
        ).append(stored)

        self._index[key] = stored

        for project_id in self._extract_project_ids(stored):
            self._project_index.setdefault(
                project_id,
                [],
            ).append(key)

    def add_messages(
        self,
        messages: Iterable[MessageEnvelope],
    ) -> None:
        """
        Последовательно добавляет сообщения.

        Если один элемент содержит duplicate key, исключение не скрывается:
        вызывающий слой должен решить, является ли это повторной доставкой,
        редактированием или реальной ошибкой.
        """

        for message in messages:
            self.add_message(message)

    # -----------------------------------------------------------------
    # READ API
    # -----------------------------------------------------------------

    def get_message(
        self,
        dialog_id: str,
        message_id: str,
    ) -> Optional[MessageEnvelope]:
        """
        Возвращает копию исходного сообщения.
        """

        item = self._index.get(
            MessageKey(
                dialog_id=dialog_id,
                message_id=message_id,
            )
        )

        return deepcopy(item) if item is not None else None

    def get_dialog_messages(
        self,
        dialog_id: str,
    ) -> List[MessageEnvelope]:
        """
        Возвращает ВСЮ сохранённую историю диалога.

        Никакого автоматического усечения нет.
        """

        return deepcopy(
            self._dialogs.get(dialog_id, [])
        )

    def get_recent_messages(
        self,
        dialog_id: str,
        limit: int = 20,
        before_message_id: Optional[str] = None,
    ) -> List[MessageEnvelope]:
        """
        Возвращает только последние сообщения для short-term context.

        Это ограничение НЕ удаляет старую историю.
        Оно действует только на результат текущего чтения.

        before_message_id:
            если задан, текущее сообщение не включается,
            выбираются сообщения ДО него.
        """

        if limit < 0:
            raise ValueError("limit не может быть отрицательным.")

        if limit == 0:
            return []

        messages = self._dialogs.get(
            dialog_id,
            [],
        )

        end = len(messages)

        if before_message_id is not None:
            end = self._find_position(
                dialog_id=dialog_id,
                message_id=before_message_id,
            )

            if end is None:
                return []

        start = max(
            0,
            end - limit,
        )

        return deepcopy(
            messages[start:end]
        )

    def get_project_messages(
        self,
        project_id: str,
        limit: Optional[int] = None,
    ) -> List[MessageEnvelope]:
        """
        Возвращает сообщения, явно связанные с проектом.

        На этом уровне используются project_id из:
        - message.projects;
        - attachments[].project;
        - links[].project.
        """

        keys = self._project_index.get(
            project_id,
            [],
        )

        if limit is not None:
            if limit < 0:
                raise ValueError("limit не может быть отрицательным.")

            if limit == 0:
                return []

            keys = keys[-limit:]

        result: List[MessageEnvelope] = []

        for key in keys:
            item = self._index.get(key)

            if item is not None:
                result.append(
                    deepcopy(item)
                )

        return result

    # -----------------------------------------------------------------
    # CONTEXT SELECTION
    # -----------------------------------------------------------------

    def build_context(
        self,
        dialog_id: str,
        current_message_id: Optional[str] = None,
        recent_limit: int = 20,
        include_reply_target: bool = True,
    ) -> DialogueContextBundle:
        """
        Формирует компактный контекст для AnalysisEngine/LLM.

        В контекст входят:
        1. recent_limit сообщений перед текущим;
        2. reply_to target, даже если он находится далеко в прошлом.

        Дубликаты удаляются.
        Полная история при этом остаётся сохранённой.
        """

        if recent_limit < 0:
            raise ValueError(
                "recent_limit не может быть отрицательным."
            )

        current = None

        if current_message_id is not None:
            current = self.get_message(
                dialog_id=dialog_id,
                message_id=current_message_id,
            )

        recent = self.get_recent_messages(
            dialog_id=dialog_id,
            limit=recent_limit,
            before_message_id=current_message_id,
        )

        reply_target = None

        if (
            include_reply_target
            and current is not None
            and current.context is not None
            and current.context.reply_to_message_id
        ):
            reply_target = self.get_message(
                dialog_id=dialog_id,
                message_id=current.context.reply_to_message_id,
            )

        selected_keys = set()
        selected: List[MessageEnvelope] = []

        def add(item: Optional[MessageEnvelope]) -> None:
            if item is None:
                return

            key = MessageKey(
                dialog_id=item.dialog_id,
                message_id=item.message_id,
            )

            if key in selected_keys:
                return

            selected_keys.add(key)
            selected.append(item)

        # Reply target включаем обязательно, даже если он старый.
        add(reply_target)

        for item in recent:
            add(item)

        # Восстанавливаем порядок исходного диалога.
        position = {
            item.message_id: idx
            for idx, item in enumerate(
                self._dialogs.get(dialog_id, [])
            )
        }

        selected.sort(
            key=lambda item: position.get(
                item.message_id,
                10**12,
            )
        )

        return DialogueContextBundle(
            dialog_id=dialog_id,
            current_message_id=current_message_id,
            messages=deepcopy(selected),
            reply_target=deepcopy(reply_target),
            recent_messages=deepcopy(recent),
        )

    # -----------------------------------------------------------------
    # COUNTERS / DIAGNOSTICS
    # -----------------------------------------------------------------

    def count_dialog(
        self,
        dialog_id: str,
    ) -> int:
        return len(
            self._dialogs.get(dialog_id, [])
        )

    def count_all(self) -> int:
        return len(self._index)

    def dialog_ids(self) -> List[str]:
        return sorted(self._dialogs)

    # -----------------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------------

    def _find_position(
        self,
        dialog_id: str,
        message_id: str,
    ) -> Optional[int]:
        messages = self._dialogs.get(
            dialog_id,
            [],
        )

        for index, item in enumerate(messages):
            if item.message_id == message_id:
                return index

        return None

    @staticmethod
    def _validate_message(
        message: MessageEnvelope,
    ) -> None:
        if not message.dialog_id:
            raise ValueError(
                "dialog_id обязателен для DialogueStore."
            )

        if not message.message_id:
            raise ValueError(
                "message_id обязателен для DialogueStore."
            )

    @staticmethod
    def _extract_project_ids(
        message: MessageEnvelope,
    ) -> List[str]:
        """
        Собирает project_id без догадок.
        """

        result: List[str] = []
        seen = set()

        def add(project) -> None:
            if project is None or not project.project_id:
                return

            if project.project_id in seen:
                return

            seen.add(project.project_id)
            result.append(project.project_id)

        for project in message.projects:
            add(project)

        for attachment in message.attachments:
            add(attachment.project)

        for link in message.links:
            add(link.project)

        return result
