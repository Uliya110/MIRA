from __future__ import annotations

import unittest

from src.assistant.analysis_schema_v2 import (
    AnalysisResult,
    ContactRole,
    MessageAnalysis,
    MessageEnvelope,
    ProjectReference,
    Requirement,
    RequirementStatus,
    Speaker,
    TaskItem,
    TaskStatus,
)
from src.assistant.project_state_engine import ProjectStateEngine


class ProjectStateEngineTests(unittest.TestCase):
    """
    Проверяет, как состояние проекта реагирует на замену требования
    и что происходит со связанными рабочими задачами.
    """

    def setUp(self) -> None:
        self.engine = ProjectStateEngine()
        self.villa_2 = ProjectReference(
            project_id="villa_2",
            project_name="Villa 2",
            confidence=0.99,
        )

    def add_initial_requirement_and_task(self) -> None:
        """Создаёт исходное требование 350 мм и связанную с ним задачу."""
        message = MessageEnvelope(
            message_id="msg_001",
            dialog_id="dialog_001",
            contact_id="architect_001",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            text="Шкаф делаем глубиной 350 мм.",
            projects=[self.villa_2],
        )

        requirement = Requirement(
            requirement_id="req_350",
            description="Глубина шкафа 350 мм",
            project=self.villa_2,
            source_message_id="msg_001",
            status=RequirementStatus.ACTIVE,
            confidence=0.99,
            entity="шкаф",
            attribute="глубина",
            value="350",
            unit="мм",
        )

        task = TaskItem(
            task_id="task_cabinet",
            description="Настроить шкаф глубиной 350 мм",
            project=self.villa_2,
            source_message_id="msg_001",
            related_requirement_ids=["req_350"],
            status=TaskStatus.NEW,
        )

        self.engine.apply(
            AnalysisResult(
                message=message,
                analysis=MessageAnalysis(
                    message_id="msg_001",
                    projects=[self.villa_2],
                    requirements=[requirement],
                    tasks=[task],
                    confidence=0.99,
                ),
                affected_project_ids=["villa_2"],
            )
        )

    def replace_requirement(self):
        """Заменяет требование 350 мм на 250 мм без добавления новой задачи."""
        message = MessageEnvelope(
            message_id="msg_002",
            dialog_id="dialog_001",
            contact_id="architect_001",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            text="Шкаф теперь 250 вместо 350.",
            projects=[self.villa_2],
        )

        requirement = Requirement(
            requirement_id="req_250",
            description="Глубина шкафа 250 мм",
            project=self.villa_2,
            source_message_id="msg_002",
            status=RequirementStatus.ACTIVE,
            confidence=0.99,
            supersedes_requirement_id="req_350",
            entity="шкаф",
            attribute="глубина",
            value="250",
            unit="мм",
        )

        return self.engine.apply(
            AnalysisResult(
                message=message,
                analysis=MessageAnalysis(
                    message_id="msg_002",
                    projects=[self.villa_2],
                    requirements=[requirement],
                    confidence=0.99,
                ),
                affected_project_ids=["villa_2"],
            )
        )

    def test_initial_requirement_and_linked_task_are_added(self) -> None:
        self.add_initial_requirement_and_task()

        state = self.engine.get_state("villa_2")

        self.assertIsNotNone(state)
        self.assertEqual(len(state.active_requirements), 1)
        self.assertEqual(
            state.active_requirements[0].requirement_id,
            "req_350",
        )
        self.assertEqual(len(state.open_tasks), 1)
        self.assertEqual(
            state.open_tasks[0].status,
            TaskStatus.NEW,
        )
        self.assertEqual(
            state.open_tasks[0].related_requirement_ids,
            ["req_350"],
        )

    def test_replacing_requirement_invalidates_linked_task(self) -> None:
        self.add_initial_requirement_and_task()
        update = self.replace_requirement()

        state = self.engine.get_state("villa_2")

        self.assertIsNotNone(state)
        self.assertFalse(
            any(
                requirement.requirement_id == "req_350"
                for requirement in state.active_requirements
            )
        )
        self.assertTrue(
            any(
                requirement.requirement_id == "req_250"
                for requirement in state.active_requirements
            )
        )

        old_task = next(
            task
            for task in state.open_tasks
            if task.task_id == "task_cabinet"
        )

        self.assertEqual(old_task.status, TaskStatus.UNCLEAR)
        self.assertTrue(old_task.requires_confirmation)
        self.assertTrue(
            any(
                change.change_type
                == "task_invalidated_by_requirement_change"
                for change in update.changes
            )
        )

    def test_new_task_can_link_to_replacement_requirement(self) -> None:
        self.add_initial_requirement_and_task()
        self.replace_requirement()

        message = MessageEnvelope(
            message_id="msg_003",
            dialog_id="dialog_001",
            contact_id="architect_001",
            speaker=Speaker.CONTACT,
            contact_role=ContactRole.ARCHITECT,
            text="Переделываем шкаф под глубину 250 мм.",
            projects=[self.villa_2],
        )

        task = TaskItem(
            task_id="task_cabinet_v2",
            description="Настроить шкаф глубиной 250 мм",
            project=self.villa_2,
            source_message_id="msg_003",
            related_requirement_ids=["req_250"],
            status=TaskStatus.NEW,
        )

        self.engine.apply(
            AnalysisResult(
                message=message,
                analysis=MessageAnalysis(
                    message_id="msg_003",
                    projects=[self.villa_2],
                    tasks=[task],
                    confidence=0.99,
                ),
                affected_project_ids=["villa_2"],
            )
        )

        state = self.engine.get_state("villa_2")
        self.assertIsNotNone(state)

        old_task = next(
            item
            for item in state.open_tasks
            if item.task_id == "task_cabinet"
        )
        new_task = next(
            item
            for item in state.open_tasks
            if item.task_id == "task_cabinet_v2"
        )

        self.assertEqual(old_task.status, TaskStatus.UNCLEAR)
        self.assertEqual(new_task.status, TaskStatus.NEW)
        self.assertEqual(
            new_task.related_requirement_ids,
            ["req_250"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
