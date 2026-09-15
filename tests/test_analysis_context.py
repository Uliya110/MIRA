from __future__ import annotations

import unittest
from datetime import datetime

from src.assistant.analysis_engine import AnalysisEngine
from src.assistant.analysis_schema_v2 import (
    AnalysisResult,
    ContactRole,
    MessageAnalysis,
    MessageContext,
    MessageEnvelope,
    ProjectReference,
    Requirement,
    RequirementStatus,
    Speaker,
)
from src.assistant.project_state_engine import ProjectStateEngine


class AnalysisContextTests(unittest.TestCase):
    """Проверяет совместную работу анализа сообщения и состояния проекта."""

    def setUp(self) -> None:
        self.state_engine = ProjectStateEngine()
        self.analysis_engine = AnalysisEngine(
            context_provider=self.state_engine,
        )
        self.project = ProjectReference(
            project_id="villa_2",
            project_name="Villa 2",
            confidence=0.99,
        )

    def add_initial_requirement(self) -> None:
        message = MessageEnvelope(
            message_id="msg_001",
            dialog_id="dialog_architect",
            contact_id="architect_001",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 10, 0),
            text="Шкаф делаем глубиной 350 мм.",
            projects=[self.project],
        )
        requirement = Requirement(
            requirement_id="req_cabinet_350",
            description="Глубина шкафа 350 мм",
            project=self.project,
            source_message_id="msg_001",
            status=RequirementStatus.ACTIVE,
            confidence=0.99,
            entity="шкаф",
            attribute="глубина",
            value="350",
            unit="мм",
        )
        self.state_engine.apply(
            AnalysisResult(
                message=message,
                analysis=MessageAnalysis(
                    message_id="msg_001",
                    projects=[self.project],
                    requirements=[requirement],
                    confidence=0.99,
                ),
                affected_project_ids=["villa_2"],
            )
        )

    def analyze_replacement(self):
        message = MessageEnvelope(
            message_id="msg_002",
            dialog_id="dialog_architect",
            contact_id="architect_001",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 11, 0),
            text="Шкаф теперь 250 мм вместо 350 мм.",
            projects=[self.project],
        )
        return self.analysis_engine.analyze(message)

    def prepare_replacement_state(self):
        self.add_initial_requirement()
        result = self.analyze_replacement()
        new_requirement = next(
            item
            for item in result.analysis.requirements
            if item.supersedes_requirement_id is not None
        )
        self.state_engine.apply(result)
        return result, new_requirement

    def test_replacement_is_resolved_from_project_state(self) -> None:
        self.add_initial_requirement()
        result = self.analyze_replacement()

        new_requirement = next(
            item
            for item in result.analysis.requirements
            if item.supersedes_requirement_id is not None
        )

        self.assertEqual(
            new_requirement.supersedes_requirement_id,
            "req_cabinet_350",
        )
        self.assertFalse(
            new_requirement.requires_human_confirmation
        )

    def test_project_state_applies_replacement(self) -> None:
        _, new_requirement = self.prepare_replacement_state()
        state = self.state_engine.get_state("villa_2")

        self.assertIsNotNone(state)
        self.assertFalse(
            any(
                item.requirement_id == "req_cabinet_350"
                for item in state.active_requirements
            )
        )
        self.assertTrue(
            any(
                item.requirement_id == new_requirement.requirement_id
                for item in state.active_requirements
            )
        )

    def test_short_reply_confirms_replied_requirement(self) -> None:
        _, new_requirement = self.prepare_replacement_state()

        reply = MessageEnvelope(
            message_id="msg_003",
            dialog_id="dialog_architect",
            contact_id="architect_001",
            speaker=Speaker.USER,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 11, 2),
            text="Поняла",
            projects=[self.project],
            context=MessageContext(
                reply_to_message_id="msg_002",
                previous_message_ids=["msg_002"],
                context_required=True,
                context_resolved=True,
            ),
        )

        result = self.analysis_engine.analyze(reply)

        self.assertEqual(len(result.analysis.decisions), 1)
        decision = result.analysis.decisions[0]
        self.assertEqual(
            decision.related_requirement_ids,
            [new_requirement.requirement_id],
        )
        self.assertTrue(decision.confirmed_by_user)

        self.state_engine.apply(result)
        state = self.state_engine.get_state("villa_2")
        accepted = next(
            item
            for item in state.active_requirements
            if item.requirement_id == new_requirement.requirement_id
        )
        self.assertEqual(
            accepted.status,
            RequirementStatus.ACCEPTED,
        )

    def test_history_records_replacement_and_acceptance(self) -> None:
        _, new_requirement = self.prepare_replacement_state()

        reply = MessageEnvelope(
            message_id="msg_003",
            dialog_id="dialog_architect",
            contact_id="architect_001",
            speaker=Speaker.USER,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 11, 2),
            text="Поняла",
            projects=[self.project],
            context=MessageContext(
                reply_to_message_id="msg_002",
                previous_message_ids=["msg_002"],
                context_required=True,
                context_resolved=True,
            ),
        )
        result = self.analysis_engine.analyze(reply)
        self.state_engine.apply(result)

        history = self.state_engine.get_history("villa_2")
        self.assertTrue(
            any(
                item.change_type == "requirement_replaced"
                for item in history
            )
        )
        self.assertTrue(
            any(
                item.change_type == "requirement_accepted"
                and item.entity_id == new_requirement.requirement_id
                for item in history
            )
        )

    def test_external_deadline_requires_user_confirmation(self) -> None:
        project = ProjectReference(
            project_id="villa_deadline",
            project_name="Villa Deadline",
            confidence=0.99,
        )
        message = MessageEnvelope(
            message_id="deadline_001",
            dialog_id="dialog_deadline",
            contact_id="client_deadline",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 13, 0),
            text="Пришли финальный рендер завтра до 12:00.",
            projects=[project],
        )

        result = self.analysis_engine.analyze(message)

        self.assertTrue(result.analysis.tasks)
        task = result.analysis.tasks[0]
        self.assertIsNotNone(task.deadline_iso)
        self.assertFalse(task.deadline_confirmed_by_user)
        self.assertTrue(result.analysis.requires_human)

    def test_short_reply_without_context_requires_human(self) -> None:
        project = ProjectReference(
            project_id="villa_unclear",
            project_name="Villa Unclear",
            confidence=0.99,
        )
        message = MessageEnvelope(
            message_id="unclear_001",
            dialog_id="dialog_unclear",
            contact_id="user",
            speaker=Speaker.USER,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 14, 0),
            text="Поняла",
            projects=[project],
        )

        result = self.analysis_engine.analyze(message)

        self.assertTrue(result.analysis.requires_human)

    def test_multiple_projects_require_clarification(self) -> None:
        project_a = ProjectReference(
            project_id="villa_a",
            project_name="Villa A",
            confidence=0.99,
        )
        project_b = ProjectReference(
            project_id="villa_b",
            project_name="Villa B",
            confidence=0.99,
        )
        message = MessageEnvelope(
            message_id="projects_001",
            dialog_id="dialog_projects",
            contact_id="architect_projects",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            timestamp=datetime(2026, 8, 30, 15, 0),
            text="Поменяй свет и пришли пререндер завтра.",
            projects=[project_a, project_b],
        )

        result = self.analysis_engine.analyze(message)

        self.assertTrue(result.analysis.requires_human)
        if result.analysis.tasks:
            self.assertTrue(
                all(task.project is None for task in result.analysis.tasks)
            )

    def test_financial_question_requires_human(self) -> None:
        project = ProjectReference(
            project_id="villa_finance",
            project_name="Villa Finance",
            confidence=0.99,
        )
        message = MessageEnvelope(
            message_id="finance_001",
            dialog_id="dialog_finance",
            contact_id="client_finance",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.CLIENT,
            timestamp=datetime(2026, 8, 30, 16, 0),
            text="Сколько будет стоить дополнительный ракурс?",
            projects=[project],
        )

        result = self.analysis_engine.analyze(message)

        self.assertTrue(result.analysis.financial_issues)
        self.assertTrue(
            result.analysis.financial_issues[0].requires_human
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
