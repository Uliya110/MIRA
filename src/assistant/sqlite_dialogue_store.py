# -*- coding: utf-8 -*-
r"""
MIRA — Persistent SQLite Dialogue Store
=======================================

Рекомендуемое расположение:
    MIRA/src/assistant/sqlite_dialogue_store.py

Назначение
----------
Постоянное хранилище полной двусторонней истории сообщений.

В отличие от in-memory DialogueStore:
- сообщения сохраняются на диск;
- после перезапуска Python история остаётся;
- число сообщений не ограничивается искусственным max_history;
- recent_limit ограничивает только контекст для модели, а не архив.

API намеренно близок к DialogueStore:
    add_message()
    add_messages()
    get_message()
    get_dialog_messages()
    get_recent_messages()
    get_project_messages()
    build_context()
    count_dialog()
    count_all()
    dialog_ids()

Это позволит позже дать AnalysisEngine общий repository protocol
и менять SQLite/PostgreSQL без переписывания NLP-логики.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from analysis_schema_v2 import (
    Attachment,
    AttachmentType,
    ContactRole,
    Link,
    LinkType,
    MessageContext,
    MessageEnvelope,
    MessageType,
    ProjectReference,
    Speaker,
)
from dialogue_store import (
    DialogueContextBundle,
    DuplicateMessageError,
)


class SQLiteDialogueStore:
    """
    Постоянное SQLite-хранилище исходных MessageEnvelope.
    """

    def __init__(
        self,
        db_path: Path | str,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._connection = sqlite3.connect(
            str(self.db_path)
        )
        self._connection.row_factory = sqlite3.Row

        # Безопасный компромисс для локального desktop-приложения:
        # WAL улучшает параллельное чтение/запись и устойчивость.
        self._connection.execute(
            "PRAGMA journal_mode=WAL;"
        )
        self._connection.execute(
            "PRAGMA foreign_keys=ON;"
        )

        self._create_schema()

    # -----------------------------------------------------------------
    # LIFECYCLE
    # -----------------------------------------------------------------

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def __enter__(self) -> "SQLiteDialogueStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -----------------------------------------------------------------
    # WRITE API
    # -----------------------------------------------------------------

    def add_message(
        self,
        message: MessageEnvelope,
    ) -> None:
        """
        Сохраняет исходное сообщение целиком.

        UNIQUE(dialog_id, message_id) защищает архив от молчаливой
        перезаписи сообщения.
        """

        self._validate_message(message)

        payload_json = self._serialize_message(
            message
        )

        project_ids = self._extract_project_ids(
            message
        )

        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    INSERT INTO messages (
                        dialog_id,
                        message_id,
                        contact_id,
                        timestamp_iso,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        message.dialog_id,
                        message.message_id,
                        message.contact_id,
                        (
                            message.timestamp.isoformat()
                            if message.timestamp is not None
                            else None
                        ),
                        payload_json,
                    ),
                )

                message_row_id = int(
                    cursor.lastrowid
                )

                for project_id in project_ids:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO message_projects (
                            message_row_id,
                            project_id
                        )
                        VALUES (?, ?)
                        """,
                        (
                            message_row_id,
                            project_id,
                        ),
                    )

        except sqlite3.IntegrityError as exc:
            raise DuplicateMessageError(
                "Сообщение уже существует: "
                f"dialog_id={message.dialog_id!r}, "
                f"message_id={message.message_id!r}"
            ) from exc

    def add_messages(
        self,
        messages: Iterable[MessageEnvelope],
    ) -> None:
        """
        Добавляет сообщения последовательно.

        Намеренно не скрывает DuplicateMessageError.
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
        row = self._connection.execute(
            """
            SELECT payload_json
            FROM messages
            WHERE dialog_id = ?
              AND message_id = ?
            """,
            (
                dialog_id,
                message_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return self._deserialize_message(
            row["payload_json"]
        )

    def get_dialog_messages(
        self,
        dialog_id: str,
    ) -> List[MessageEnvelope]:
        """
        Возвращает ВСЮ историю диалога в порядке сохранения.
        """

        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM messages
            WHERE dialog_id = ?
            ORDER BY id ASC
            """,
            (dialog_id,),
        ).fetchall()

        return [
            self._deserialize_message(
                row["payload_json"]
            )
            for row in rows
        ]

    def get_recent_messages(
        self,
        dialog_id: str,
        limit: int = 20,
        before_message_id: Optional[str] = None,
    ) -> List[MessageEnvelope]:
        """
        Возвращает short-term context.

        limit НЕ удаляет и НЕ изменяет историю на диске.
        """

        if limit < 0:
            raise ValueError(
                "limit не может быть отрицательным."
            )

        if limit == 0:
            return []

        before_row_id = None

        if before_message_id is not None:
            row = self._connection.execute(
                """
                SELECT id
                FROM messages
                WHERE dialog_id = ?
                  AND message_id = ?
                """,
                (
                    dialog_id,
                    before_message_id,
                ),
            ).fetchone()

            if row is None:
                return []

            before_row_id = int(row["id"])

        if before_row_id is None:
            rows = self._connection.execute(
                """
                SELECT id, payload_json
                FROM messages
                WHERE dialog_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    dialog_id,
                    limit,
                ),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT id, payload_json
                FROM messages
                WHERE dialog_id = ?
                  AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    dialog_id,
                    before_row_id,
                    limit,
                ),
            ).fetchall()

        # SQL выбирает последние N в обратном порядке.
        rows = list(reversed(rows))

        return [
            self._deserialize_message(
                row["payload_json"]
            )
            for row in rows
        ]

    def get_project_messages(
        self,
        project_id: str,
        limit: Optional[int] = None,
    ) -> List[MessageEnvelope]:
        """
        Возвращает сообщения, явно проиндексированные по project_id.
        """

        if limit is not None and limit < 0:
            raise ValueError(
                "limit не может быть отрицательным."
            )

        if limit == 0:
            return []

        if limit is None:
            rows = self._connection.execute(
                """
                SELECT m.id, m.payload_json
                FROM messages AS m
                JOIN message_projects AS mp
                  ON mp.message_row_id = m.id
                WHERE mp.project_id = ?
                ORDER BY m.id ASC
                """,
                (project_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT id, payload_json
                FROM (
                    SELECT m.id, m.payload_json
                    FROM messages AS m
                    JOIN message_projects AS mp
                      ON mp.message_row_id = m.id
                    WHERE mp.project_id = ?
                    ORDER BY m.id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (
                    project_id,
                    limit,
                ),
            ).fetchall()

        return [
            self._deserialize_message(
                row["payload_json"]
            )
            for row in rows
        ]

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
        Формирует контекст:
        - последние recent_limit сообщений ДО current;
        - reply_to target независимо от его возраста.
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

        # Собираем id выбранных сообщений.
        selected_ids = {
            item.message_id
            for item in recent
        }

        if reply_target is not None:
            selected_ids.add(
                reply_target.message_id
            )

        if not selected_ids:
            selected = []
        else:
            placeholders = ",".join(
                "?"
                for _ in selected_ids
            )

            params = [
                dialog_id,
                *selected_ids,
            ]

            rows = self._connection.execute(
                f"""
                SELECT payload_json
                FROM messages
                WHERE dialog_id = ?
                  AND message_id IN ({placeholders})
                ORDER BY id ASC
                """,
                params,
            ).fetchall()

            selected = [
                self._deserialize_message(
                    row["payload_json"]
                )
                for row in rows
            ]

        return DialogueContextBundle(
            dialog_id=dialog_id,
            current_message_id=current_message_id,
            messages=selected,
            reply_target=reply_target,
            recent_messages=recent,
        )

    # -----------------------------------------------------------------
    # COUNTERS / DIAGNOSTICS
    # -----------------------------------------------------------------

    def count_dialog(
        self,
        dialog_id: str,
    ) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS count_value
            FROM messages
            WHERE dialog_id = ?
            """,
            (dialog_id,),
        ).fetchone()

        return int(row["count_value"])

    def count_all(self) -> int:
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS count_value
            FROM messages
            """
        ).fetchone()

        return int(row["count_value"])

    def dialog_ids(self) -> List[str]:
        rows = self._connection.execute(
            """
            SELECT DISTINCT dialog_id
            FROM messages
            ORDER BY dialog_id ASC
            """
        ).fetchall()

        return [
            str(row["dialog_id"])
            for row in rows
        ]

    # -----------------------------------------------------------------
    # SCHEMA
    # -----------------------------------------------------------------

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    contact_id TEXT NOT NULL,
                    timestamp_iso TEXT,
                    payload_json TEXT NOT NULL,
                    UNIQUE(dialog_id, message_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_dialog_id
                    ON messages(dialog_id, id);

                CREATE TABLE IF NOT EXISTS message_projects (
                    message_row_id INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    PRIMARY KEY(message_row_id, project_id),
                    FOREIGN KEY(message_row_id)
                        REFERENCES messages(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_message_projects_project
                    ON message_projects(project_id, message_row_id);
                """
            )

    # -----------------------------------------------------------------
    # SERIALIZATION
    # -----------------------------------------------------------------

    @staticmethod
    def _serialize_message(
        message: MessageEnvelope,
    ) -> str:
        payload = asdict(message)

        if message.timestamp is not None:
            payload["timestamp"] = (
                message.timestamp.isoformat()
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize_message(
        raw_json: str,
    ) -> MessageEnvelope:
        payload = json.loads(raw_json)

        timestamp = payload.get("timestamp")

        if timestamp:
            from datetime import datetime
            timestamp = datetime.fromisoformat(
                timestamp
            )

        projects = [
            ProjectReference(**item)
            for item in payload.get(
                "projects",
                [],
            )
        ]

        attachments = []

        for item in payload.get(
            "attachments",
            [],
        ):
            item = dict(item)

            project_payload = item.get(
                "project"
            )

            item["project"] = (
                ProjectReference(
                    **project_payload
                )
                if project_payload
                else None
            )

            item["attachment_type"] = AttachmentType(
                item["attachment_type"]
            )

            attachments.append(
                Attachment(**item)
            )

        links = []

        for item in payload.get(
            "links",
            [],
        ):
            item = dict(item)

            project_payload = item.get(
                "project"
            )

            item["project"] = (
                ProjectReference(
                    **project_payload
                )
                if project_payload
                else None
            )

            item["link_type"] = LinkType(
                item["link_type"]
            )

            links.append(
                Link(**item)
            )

        context_payload = payload.get(
            "context"
        )

        context = (
            MessageContext(
                **context_payload
            )
            if context_payload
            else MessageContext()
        )

        return MessageEnvelope(
            message_id=payload["message_id"],
            dialog_id=payload["dialog_id"],
            contact_id=payload["contact_id"],
            speaker=Speaker(
                payload["speaker"]
            ),
            contact_role=ContactRole(
                payload.get(
                    "contact_role",
                    ContactRole.UNKNOWN.value,
                )
            ),
            timestamp=timestamp,
            message_type=MessageType(
                payload.get(
                    "message_type",
                    MessageType.TEXT.value,
                )
            ),
            text=payload.get(
                "text",
                "",
            ),
            attachments=attachments,
            links=links,
            projects=projects,
            context=context,
        )

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_message(
        message: MessageEnvelope,
    ) -> None:
        if not message.dialog_id:
            raise ValueError(
                "dialog_id обязателен."
            )

        if not message.message_id:
            raise ValueError(
                "message_id обязателен."
            )

    @staticmethod
    def _extract_project_ids(
        message: MessageEnvelope,
    ) -> List[str]:
        result: List[str] = []
        seen = set()

        def add(project) -> None:
            if (
                project is None
                or not project.project_id
                or project.project_id in seen
            ):
                return

            seen.add(project.project_id)
            result.append(
                project.project_id
            )

        for project in message.projects:
            add(project)

        for attachment in message.attachments:
            add(attachment.project)

        for link in message.links:
            add(link.project)

        return result
