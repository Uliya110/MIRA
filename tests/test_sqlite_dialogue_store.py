from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.assistant.analysis_schema_v2 import (
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
from src.assistant.dialogue_store import DuplicateMessageError
from src.assistant.sqlite_dialogue_store import SQLiteDialogueStore


class SQLiteDialogueStoreTests(unittest.TestCase):
    """Проверяет постоянное хранение истории MIRA в SQLite."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="mira_dialogue_store_"
        )
        self.db_path = Path(self._temp_dir.name) / "dialogues.sqlite3"
        self.store = SQLiteDialogueStore(self.db_path)

        self.project = ProjectReference(
            project_id="project_a",
            project_name="Project A",
            confidence=0.99,
        )
        self.base = datetime(2026, 9, 8, 9, 0)

    def tearDown(self) -> None:
        if getattr(self, "store", None) is not None:
            try:
                self.store.close()
            except Exception:
                pass
        self._temp_dir.cleanup()

    def add_messages(self, count: int) -> None:
        for index in range(1, count + 1):
            self.store.add_message(
                MessageEnvelope(
                    message_id=f"m_{index:03d}",
                    dialog_id="dialog_a",
                    contact_id="client_a",
                    speaker=(
                        Speaker.CONTACT
                        if index % 2
                        else Speaker.USER
                    ),
                    contact_role=ContactRole.CLIENT,
                    timestamp=self.base + timedelta(minutes=index),
                    text=f"Сообщение {index}",
                    projects=[self.project],
                )
            )

    def add_rich_message(self) -> MessageEnvelope:
        message = MessageEnvelope(
            message_id="m_151",
            dialog_id="dialog_a",
            contact_id="client_a",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=self.base + timedelta(minutes=151),
            message_type=MessageType.MIXED,
            text="Вот файл и ссылка.",
            attachments=[
                Attachment(
                    attachment_id="att_001",
                    attachment_type=AttachmentType.PDF,
                    file_name="brief.pdf",
                    mime_type="application/pdf",
                    description="Техническое задание",
                    project=self.project,
                )
            ],
            links=[
                Link(
                    link_id="link_001",
                    url="https://example.com/reference",
                    link_type=LinkType.REFERENCE,
                    description="Референс",
                    project=self.project,
                    source_message_id="m_151",
                )
            ],
            projects=[self.project],
        )
        self.store.add_message(message)
        return message

    def reopen_store(self) -> None:
        self.store.close()
        self.store = SQLiteDialogueStore(self.db_path)

    def test_full_history_is_written_to_sqlite(self) -> None:
        self.add_messages(150)
        self.assertEqual(self.store.count_dialog("dialog_a"), 150)

    def test_message_with_attachment_and_link_is_restored(self) -> None:
        message = self.add_rich_message()
        restored = self.store.get_message("dialog_a", "m_151")

        self.assertIsNotNone(restored)
        self.assertEqual(restored.text, message.text)
        self.assertEqual(restored.message_type, MessageType.MIXED)
        self.assertEqual(len(restored.attachments), 1)
        self.assertEqual(restored.attachments[0].file_name, "brief.pdf")
        self.assertEqual(
            restored.attachments[0].attachment_type,
            AttachmentType.PDF,
        )
        self.assertEqual(len(restored.links), 1)
        self.assertEqual(restored.links[0].link_type, LinkType.REFERENCE)

    def test_history_survives_close_and_reopen(self) -> None:
        self.add_messages(150)
        self.add_rich_message()
        self.reopen_store()

        self.assertEqual(self.store.count_dialog("dialog_a"), 151)

        first = self.store.get_message("dialog_a", "m_001")
        self.assertIsNotNone(first)
        self.assertEqual(first.text, "Сообщение 1")

        restored = self.store.get_message("dialog_a", "m_151")
        self.assertIsNotNone(restored)
        self.assertEqual(len(restored.attachments), 1)
        self.assertEqual(len(restored.links), 1)

    def test_recent_window_does_not_truncate_sqlite_archive(self) -> None:
        self.add_messages(150)
        self.add_rich_message()
        self.reopen_store()

        recent = self.store.get_recent_messages("dialog_a", limit=5)

        self.assertEqual(len(recent), 5)
        self.assertEqual(self.store.count_dialog("dialog_a"), 151)

    def test_old_reply_target_is_available_after_reopen(self) -> None:
        self.add_messages(150)
        self.add_rich_message()
        self.reopen_store()

        current = MessageEnvelope(
            message_id="m_152",
            dialog_id="dialog_a",
            contact_id="client_a",
            speaker=Speaker.USER,
            contact_role=ContactRole.CLIENT,
            timestamp=self.base + timedelta(minutes=152),
            text="Да.",
            projects=[self.project],
            context=MessageContext(
                reply_to_message_id="m_002",
                previous_message_ids=["m_151"],
                context_required=True,
                context_resolved=True,
            ),
        )
        self.store.add_message(current)

        bundle = self.store.build_context(
            dialog_id="dialog_a",
            current_message_id="m_152",
            recent_limit=5,
        )
        context_ids = [item.message_id for item in bundle.messages]

        self.assertIn("m_002", context_ids)
        self.assertIsNotNone(bundle.reply_target)
        self.assertEqual(bundle.reply_target.message_id, "m_002")

    def test_project_index_survives_reopen(self) -> None:
        self.add_messages(150)
        self.add_rich_message()
        self.reopen_store()

        project_messages = self.store.get_project_messages("project_a")

        self.assertEqual(len(project_messages), 151)
        self.assertTrue(
            all(
                any(
                    project.project_id == "project_a"
                    for project in message.projects
                )
                for message in project_messages
            )
        )

    def test_duplicate_message_is_blocked_after_reopen(self) -> None:
        self.add_messages(1)
        self.reopen_store()

        with self.assertRaises(DuplicateMessageError):
            self.store.add_message(
                MessageEnvelope(
                    message_id="m_001",
                    dialog_id="dialog_a",
                    contact_id="client_a",
                    speaker=Speaker.CONTACT,
                    contact_role=ContactRole.CLIENT,
                    text="Попытка перезаписи",
                    projects=[self.project],
                )
            )

        original = self.store.get_message("dialog_a", "m_001")
        self.assertIsNotNone(original)
        self.assertEqual(original.text, "Сообщение 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
