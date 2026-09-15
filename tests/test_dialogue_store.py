from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from src.assistant.analysis_schema_v2 import (
    ContactRole,
    MessageContext,
    MessageEnvelope,
    ProjectReference,
    Speaker,
)
from src.assistant.dialogue_store import (
    DialogueStore,
    DuplicateMessageError,
)


class DialogueStoreTests(unittest.TestCase):
    """
    Проверяет полное хранилище переписки MIRA и выбор контекста.
    """

    def setUp(self) -> None:
        self.store = DialogueStore()

        self.project_a = ProjectReference(
            project_id="project_a",
            project_name="Project A",
            confidence=0.99,
        )
        self.project_b = ProjectReference(
            project_id="project_b",
            project_name="Project B",
            confidence=0.99,
        )

        self.base = datetime(2026, 9, 8, 9, 0)

    def add_dialog_a_messages(self, count: int = 120) -> None:
        for index in range(1, count + 1):
            self.store.add_message(
                MessageEnvelope(
                    message_id=f"a_{index:03d}",
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
                    projects=[self.project_a],
                )
            )

    def test_full_history_is_not_truncated(self) -> None:
        self.add_dialog_a_messages(120)

        self.assertEqual(
            self.store.count_dialog("dialog_a"),
            120,
        )
        self.assertEqual(
            len(self.store.get_dialog_messages("dialog_a")),
            120,
        )

    def test_recent_window_does_not_remove_archive(self) -> None:
        self.add_dialog_a_messages(120)

        recent = self.store.get_recent_messages(
            dialog_id="dialog_a",
            limit=5,
        )

        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[0].message_id, "a_116")
        self.assertEqual(recent[-1].message_id, "a_120")
        self.assertEqual(
            self.store.count_dialog("dialog_a"),
            120,
        )

    def test_old_reply_target_is_included_in_small_context(self) -> None:
        self.add_dialog_a_messages(120)

        current = MessageEnvelope(
            message_id="a_121",
            dialog_id="dialog_a",
            contact_id="client_a",
            speaker=Speaker.USER,
            contact_role=ContactRole.CLIENT,
            timestamp=self.base + timedelta(minutes=121),
            text="Да.",
            projects=[self.project_a],
            context=MessageContext(
                reply_to_message_id="a_002",
                previous_message_ids=["a_120"],
                context_required=True,
                context_resolved=True,
            ),
        )
        self.store.add_message(current)

        bundle = self.store.build_context(
            dialog_id="dialog_a",
            current_message_id="a_121",
            recent_limit=5,
            include_reply_target=True,
        )

        context_ids = [
            message.message_id
            for message in bundle.messages
        ]

        self.assertIsNotNone(bundle.reply_target)
        self.assertEqual(
            bundle.reply_target.message_id,
            "a_002",
        )
        self.assertIn("a_002", context_ids)

        for message_id in {
            "a_116",
            "a_117",
            "a_118",
            "a_119",
            "a_120",
        }:
            self.assertIn(message_id, context_ids)

        self.assertEqual(
            self.store.count_dialog("dialog_a"),
            121,
        )

    def test_dialogues_are_isolated(self) -> None:
        self.add_dialog_a_messages(3)

        self.store.add_message(
            MessageEnvelope(
                message_id="b_001",
                dialog_id="dialog_b",
                contact_id="client_b",
                speaker=Speaker.CONTACT,
                contact_role=ContactRole.CLIENT,
                timestamp=self.base,
                text="Сообщение другого диалога",
                projects=[self.project_b],
            )
        )

        self.assertEqual(
            self.store.count_dialog("dialog_b"),
            1,
        )
        self.assertTrue(
            all(
                item.dialog_id == "dialog_a"
                for item in self.store.get_dialog_messages("dialog_a")
            )
        )
        self.assertTrue(
            all(
                item.dialog_id == "dialog_b"
                for item in self.store.get_dialog_messages("dialog_b")
            )
        )

    def test_messages_can_be_selected_by_project(self) -> None:
        self.add_dialog_a_messages(5)

        self.store.add_message(
            MessageEnvelope(
                message_id="b_001",
                dialog_id="dialog_b",
                contact_id="client_b",
                speaker=Speaker.CONTACT,
                contact_role=ContactRole.CLIENT,
                timestamp=self.base,
                text="Сообщение другого проекта",
                projects=[self.project_b],
            )
        )

        project_a_messages = self.store.get_project_messages(
            "project_a"
        )
        project_b_messages = self.store.get_project_messages(
            "project_b"
        )

        self.assertEqual(len(project_a_messages), 5)
        self.assertEqual(len(project_b_messages), 1)
        self.assertTrue(
            all(
                item.projects[0].project_id == "project_a"
                for item in project_a_messages
            )
        )
        self.assertTrue(
            all(
                item.projects[0].project_id == "project_b"
                for item in project_b_messages
            )
        )

    def test_returned_copy_cannot_modify_saved_archive(self) -> None:
        self.add_dialog_a_messages(1)

        copied = self.store.get_message(
            dialog_id="dialog_a",
            message_id="a_001",
        )
        self.assertIsNotNone(copied)

        original_text = copied.text
        copied.text = "ИСПОРЧЕНО СНАРУЖИ"

        stored_again = self.store.get_message(
            dialog_id="dialog_a",
            message_id="a_001",
        )

        self.assertIsNotNone(stored_again)
        self.assertEqual(
            stored_again.text,
            original_text,
        )

    def test_duplicate_message_does_not_overwrite_history(self) -> None:
        self.add_dialog_a_messages(1)

        with self.assertRaises(DuplicateMessageError):
            self.store.add_message(
                MessageEnvelope(
                    message_id="a_001",
                    dialog_id="dialog_a",
                    contact_id="client_a",
                    speaker=Speaker.CONTACT,
                    contact_role=ContactRole.CLIENT,
                    timestamp=self.base,
                    text="Попытка перезаписи",
                    projects=[self.project_a],
                )
            )

        stored = self.store.get_message(
            dialog_id="dialog_a",
            message_id="a_001",
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored.text, "Сообщение 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
