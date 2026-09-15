from __future__ import annotations

import unittest
from datetime import datetime

from src.assistant.analysis_schema_v2 import (
    Agreement,
    AnalysisResult,
    ContactRole,
    MessageAnalysis,
    MessageEnvelope,
    MessageEvent,
    MessageEventType,
    ProjectReference,
    Requirement,
    RequirementStatus,
    ReminderKind,
    ReminderStatus,
    Speaker,
    TaskItem,
    TaskStatus,
)
from src.assistant.project_state_engine import ProjectStateEngine


class ProjectHistoryReminderTests(unittest.TestCase):
    """
    Проверяет историю проекта и создание/отмену напоминаний MIRA.
    """

    def setUp(self) -> None:
        self.engine = ProjectStateEngine()
        self.project = ProjectReference(
            project_id="villa_2",
            project_name="Villa 2",
            confidence=0.99,
        )

    def apply_message(
        self,
        message: MessageEnvelope,
        analysis: MessageAnalysis,
    ):
        return self.engine.apply(
            AnalysisResult(
                message=message,
                analysis=analysis,
                affected_project_ids=["villa_2"],
            )
        )

    def add_replaced_requirement(self) -> None:
        first_message = MessageEnvelope(
            message_id="m1",
            dialog_id="d1",
            contact_id="architect",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 10, 0),
            text="Шкаф 350 мм.",
            projects=[self.project],
        )
        old_requirement = Requirement(
            requirement_id="req350",
            description="Глубина шкафа 350 мм",
            project=self.project,
            source_message_id="m1",
            status=RequirementStatus.ACTIVE,
            confidence=0.99,
        )
        self.apply_message(
            first_message,
            MessageAnalysis(
                message_id="m1",
                projects=[self.project],
                requirements=[old_requirement],
                confidence=0.99,
            ),
        )

        second_message = MessageEnvelope(
            message_id="m2",
            dialog_id="d1",
            contact_id="architect",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 11, 0),
            text="Шкаф 250 вместо 350.",
            projects=[self.project],
        )
        new_requirement = Requirement(
            requirement_id="req250",
            description="Глубина шкафа 250 мм",
            project=self.project,
            source_message_id="m2",
            status=RequirementStatus.ACTIVE,
            confidence=0.99,
            supersedes_requirement_id="req350",
        )
        self.apply_message(
            second_message,
            MessageAnalysis(
                message_id="m2",
                projects=[self.project],
                requirements=[new_requirement],
                confidence=0.99,
            ),
        )

    def add_task_with_deadline(self) -> None:
        message = MessageEnvelope(
            message_id="m3",
            dialog_id="d1",
            contact_id="client",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 12, 0),
            text="Пришли финал завтра к 18:00.",
            projects=[self.project],
        )
        task = TaskItem(
            task_id="task_final",
            description="Отправить финальный рендер",
            project=self.project,
            deadline_text="завтра к 18:00",
            deadline_iso="2026-08-31T18:00:00",
            source_message_id="m3",
            status=TaskStatus.NEW,
        )
        self.apply_message(
            message,
            MessageAnalysis(
                message_id="m3",
                projects=[self.project],
                tasks=[task],
                confidence=0.99,
            ),
        )

    def add_active_agreement(self) -> None:
        message = MessageEnvelope(
            message_id="m4",
            dialog_id="d1",
            contact_id="client",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 13, 0),
            text="В понедельник согласуем второй вариант.",
            projects=[self.project],
        )
        agreement = Agreement(
            agreement_id="agr_review",
            description="Согласовать второй вариант",
            project=self.project,
            active=True,
            deadline_text="в понедельник",
            deadline_iso="2026-08-31T12:00:00",
            reminder_required=True,
            source_message_ids=["m4"],
            confirmed_by=["client"],
        )
        self.apply_message(
            message,
            MessageAnalysis(
                message_id="m4",
                projects=[self.project],
                agreements=[agreement],
                confidence=0.99,
            ),
        )

    def add_scheduled_event(self) -> None:
        message = MessageEnvelope(
            message_id="m5",
            dialog_id="d1",
            contact_id="client",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 14, 0),
            text="Во вторник созвон в 11.",
            projects=[self.project],
        )
        event = MessageEvent(
            event_id="event_call",
            event_type=MessageEventType.OTHER,
            project=self.project,
            source_message_id="m5",
            description="Созвон с клиентом",
            scheduled_for_text="во вторник в 11:00",
            scheduled_for_iso="2026-09-01T11:00:00",
            reminder_required=True,
            confidence=0.99,
        )
        self.apply_message(
            message,
            MessageAnalysis(
                message_id="m5",
                projects=[self.project],
                events=[event],
                confidence=0.99,
            ),
        )

    def cancel_agreement(self) -> None:
        message = MessageEnvelope(
            message_id="m6",
            dialog_id="d1",
            contact_id="client",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 15, 0),
            text="В понедельник уже не согласуем, отменяем.",
            projects=[self.project],
        )
        agreement = Agreement(
            agreement_id="agr_review",
            description="Согласовать второй вариант",
            project=self.project,
            active=False,
            source_message_ids=["m6"],
            confirmed_by=["client"],
        )
        self.apply_message(
            message,
            MessageAnalysis(
                message_id="m6",
                projects=[self.project],
                agreements=[agreement],
                confidence=0.99,
            ),
        )

    def test_history_keeps_requirement_replacement(self) -> None:
        self.add_replaced_requirement()

        history = self.engine.get_history("villa_2")

        self.assertTrue(
            any(
                item.change_type == "requirement_replaced"
                for item in history
            )
        )

    def test_current_state_keeps_only_new_requirement(self) -> None:
        self.add_replaced_requirement()

        state = self.engine.get_state("villa_2")

        self.assertIsNotNone(state)
        self.assertEqual(len(state.active_requirements), 1)
        self.assertEqual(
            state.active_requirements[0].requirement_id,
            "req250",
        )

    def test_task_deadline_creates_reminder_one_hour_before(self) -> None:
        self.add_task_with_deadline()

        reminder = next(
            item
            for item in self.engine.get_reminders("villa_2")
            if item.kind == ReminderKind.TASK_DEADLINE
        )

        self.assertTrue(
            reminder.due_at_iso.startswith("2026-08-31T18:00")
        )
        self.assertTrue(
            reminder.trigger_at_iso.startswith("2026-08-31T17:00")
        )

    def test_agreement_deadline_creates_reminder(self) -> None:
        self.add_active_agreement()

        reminders = self.engine.get_reminders("villa_2")

        self.assertTrue(
            any(
                item.kind == ReminderKind.AGREEMENT_DEADLINE
                for item in reminders
            )
        )

    def test_scheduled_event_creates_reminder(self) -> None:
        self.add_scheduled_event()

        reminders = self.engine.get_reminders("villa_2")

        self.assertTrue(
            any(
                item.kind == ReminderKind.SCHEDULED_EVENT
                for item in reminders
            )
        )

    def test_cancelling_agreement_cancels_its_reminder(self) -> None:
        self.add_active_agreement()
        self.cancel_agreement()

        agreement_reminders = [
            item
            for item in self.engine.get_reminders(
                "villa_2",
                pending_only=False,
            )
            if item.source_entity_id == "agr_review"
        ]

        self.assertTrue(agreement_reminders)
        self.assertTrue(
            all(
                item.status == ReminderStatus.CANCELLED
                for item in agreement_reminders
            )
        )

    def test_user_reminder_texts_are_in_russian(self) -> None:
        self.add_task_with_deadline()
        self.add_active_agreement()
        self.add_scheduled_event()

        pending = self.engine.get_reminders("villa_2")

        self.assertTrue(pending)
        for reminder in pending:
            self.assertIn(
                "Напоминание по проекту",
                reminder.user_message_ru,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
