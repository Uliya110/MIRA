from __future__ import annotations

import unittest

from src.assistant.analysis_schema_v2 import (
    AnalysisResult,
    Attachment,
    AttachmentType,
    ConflictAssessment,
    ConflictLevel,
    ContactRole,
    Decision,
    DecisionStatus,
    FinancialIssue,
    IntentItem,
    IntentType,
    Link,
    LinkType,
    MessageAnalysis,
    MessageContext,
    MessageEnvelope,
    MessageEvent,
    MessageEventType,
    MessageType,
    PriorityLevel,
    ProjectReference,
    Requirement,
    RequirementStatus,
    Speaker,
    TaskItem,
    TaskStatus,
    ValidationStatus,
    VisualRequirementCheck,
    validate_analysis_result,
)


def project(project_id: str, name: str, confidence: float = 0.99) -> ProjectReference:
    """Создаёт ссылку на проект для тестовых сценариев."""
    return ProjectReference(
        project_id=project_id,
        project_name=name,
        confidence=confidence,
    )


class AnalysisSchemaTests(unittest.TestCase):
    """Проверка основных сценариев схемы данных MIRA версии 2.4."""

    def assert_valid(self, result: AnalysisResult) -> None:
        errors = validate_analysis_result(result)
        self.assertEqual(
            errors,
            [],
            msg="Схема не прошла валидацию:\n" + "\n".join(errors),
        )

    def setUp(self) -> None:
        self.villa_2 = project("villa_2", "Villa 2")
        self.villa_5 = project("villa_5", "Villa 5")
        self.villa_8 = project("villa_8", "Villa 8")

    def test_requirement_replacement(self) -> None:
        message = MessageEnvelope(
            "m1",
            "d1",
            "architect",
            Speaker.CONTACT,
            ContactRole.ARCHITECT,
            text="Шкаф 250 вместо 350.",
            projects=[self.villa_2],
        )

        analysis = MessageAnalysis(
            message_id="m1",
            projects=[self.villa_2],
            intents=[
                IntentItem(IntentType.EDITS, 0.99, self.villa_2)
            ],
            requirements=[
                Requirement(
                    "req250",
                    "Глубина шкафа 250 мм",
                    self.villa_2,
                    "m1",
                    [],
                    [],
                    RequirementStatus.ACTIVE,
                    0.99,
                    "req350",
                    "шкаф",
                    "глубина",
                    "250",
                    "мм",
                    False,
                )
            ],
            tasks=[
                TaskItem(
                    "task1",
                    "Изменить глубину шкафа",
                    self.villa_2,
                    status=TaskStatus.NEW,
                )
            ],
            confidence=0.99,
        )

        result = AnalysisResult(
            message,
            analysis,
            ["villa_2"],
        )
        self.assert_valid(result)

    def test_short_reply_requires_context(self) -> None:
        message = MessageEnvelope(
            "m2",
            "d1",
            "architect",
            Speaker.USER,
            ContactRole.ARCHITECT,
            text="Поняла",
            projects=[self.villa_2],
            context=MessageContext("m1", ["m1"], True, True),
        )

        analysis = MessageAnalysis(
            message_id="m2",
            projects=[self.villa_2],
            intents=[
                IntentItem(IntentType.APPROVAL, 0.96, self.villa_2)
            ],
            decisions=[
                Decision(
                    "dec1",
                    "Правка принята",
                    self.villa_2,
                    ["req250"],
                    "m2",
                    [],
                    [],
                    DecisionStatus.CONFIRMED,
                    True,
                    0.96,
                    True,
                )
            ],
            events=[
                MessageEvent(
                    "ev1",
                    MessageEventType.TASK_ACCEPTED,
                    self.villa_2,
                    "m2",
                    description="Пользователь подтвердил правку",
                    confidence=0.96,
                )
            ],
            confidence=0.96,
        )

        result = AnalysisResult(
            message,
            analysis,
            ["villa_2"],
        )
        self.assert_valid(result)

    def test_multiple_projects_in_one_message(self) -> None:
        message = MessageEnvelope(
            "m3",
            "d2",
            "manager",
            Speaker.CONTACT,
            ContactRole.MANAGER,
            text=(
                "По второй переделай гардероб. "
                "По пятой экстерьер пока не делай. "
                "Восьмую пришли сегодня."
            ),
            projects=[self.villa_2, self.villa_5, self.villa_8],
        )

        analysis = MessageAnalysis(
            message_id="m3",
            projects=[self.villa_2, self.villa_5, self.villa_8],
            intents=[
                IntentItem(IntentType.EDITS, 0.99, self.villa_2),
                IntentItem(IntentType.PRIORITY_CHANGE, 0.98, self.villa_5),
                IntentItem(IntentType.RESULT_DELIVERY, 0.97, self.villa_8),
                IntentItem(IntentType.DEADLINE, 0.95, self.villa_8),
            ],
            tasks=[
                TaskItem(
                    "t_v2",
                    "Переделать гардероб",
                    self.villa_2,
                    status=TaskStatus.NEW,
                ),
                TaskItem(
                    "t_v5",
                    "Приостановить экстерьер",
                    self.villa_5,
                    status=TaskStatus.PAUSED,
                ),
                TaskItem(
                    "t_v8",
                    "Отправить финал",
                    self.villa_8,
                    deadline_text="сегодня",
                    status=TaskStatus.NEW,
                    requires_confirmation=True,
                ),
            ],
            urgency=PriorityLevel.HIGH,
            requires_human=True,
            human_reason="Срок должен быть подтверждён пользователем.",
            confidence=0.97,
        )

        result = AnalysisResult(
            message,
            analysis,
            ["villa_2", "villa_5", "villa_8"],
        )
        self.assert_valid(result)

    def test_link_without_explanation(self) -> None:
        unknown = ProjectReference(
            None,
            None,
            0.2,
            True,
            True,
        )

        link = Link(
            "link1",
            "https://example.com/x",
            LinkType.UNKNOWN,
            "Ссылка без пояснения",
            None,
            unknown,
            "m4",
            False,
            0.2,
            False,
            None,
        )

        message = MessageEnvelope(
            "m4",
            "d3",
            "client",
            Speaker.CONTACT,
            ContactRole.CLIENT,
            message_type=MessageType.MIXED,
            text="Вот",
            links=[link],
            projects=[unknown],
            context=MessageContext(None, [], True, False),
        )

        analysis = MessageAnalysis(
            message_id="m4",
            projects=[unknown],
            intents=[
                IntentItem(IntentType.UNKNOWN, 0.2, unknown)
            ],
            requires_human=True,
            human_reason="Неясно назначение ссылки и проект.",
            recommendation="Проверить контекст или уточнить.",
            confidence=0.2,
        )

        result = AnalysisResult(
            message,
            analysis,
            [],
        )
        self.assert_valid(result)

    def test_multiple_financial_facts(self) -> None:
        project_a = project("project_a", "Project A")

        message = MessageEnvelope(
            "m5",
            "d4",
            "client2",
            Speaker.USER,
            ContactRole.CLIENT,
            text=(
                "Ракурс 1500, моделирование 3000, "
                "два прошлых ракурса оплачены."
            ),
            projects=[project_a],
        )

        analysis = MessageAnalysis(
            message_id="m5",
            projects=[project_a],
            intents=[
                IntentItem(IntentType.COST, 0.99, project_a),
                IntentItem(IntentType.PAYMENT, 0.97, project_a),
            ],
            financial_issues=[
                FinancialIssue(
                    True,
                    "дополнительный ракурс",
                    1500,
                    "RUB",
                    "confirmed",
                    project_a,
                    False,
                ),
                FinancialIssue(
                    True,
                    "моделирование",
                    3000,
                    "RUB",
                    "confirmed",
                    project_a,
                    False,
                ),
                FinancialIssue(
                    True,
                    "два прошлых ракурса",
                    None,
                    None,
                    "paid",
                    project_a,
                    False,
                ),
            ],
            confidence=0.98,
        )

        result = AnalysisResult(
            message,
            analysis,
            ["project_a"],
        )
        self.assert_valid(result)

    def test_preview_and_visual_validation(self) -> None:
        attachment = Attachment(
            "render1",
            AttachmentType.IMAGE,
            "preview.jpg",
            description="Пререндер",
            content_analyzed=True,
            analysis_confidence=0.9,
            project=self.villa_5,
        )

        message = MessageEnvelope(
            "m6",
            "d1",
            "architect",
            Speaker.USER,
            ContactRole.ARCHITECT,
            message_type=MessageType.IMAGE,
            attachments=[attachment],
            projects=[self.villa_5],
        )

        analysis = MessageAnalysis(
            message_id="m6",
            projects=[self.villa_5],
            intents=[
                IntentItem(
                    IntentType.RESULT_DELIVERY,
                    0.98,
                    self.villa_5,
                )
            ],
            events=[
                MessageEvent(
                    "ev_preview",
                    MessageEventType.PREVIEW_SENT,
                    self.villa_5,
                    "m6",
                    ["render1"],
                    [],
                    "Отправлен пререндер",
                    0.98,
                    False,
                )
            ],
            confidence=0.97,
        )

        checks = [
            VisualRequirementCheck(
                "req_mirror",
                "Увеличить зеркало",
                "render1",
                ValidationStatus.MATCH,
                0.87,
                "Визуально похоже на выполненное.",
                True,
            ),
            VisualRequirementCheck(
                "req_glass",
                "Стекло 2400 мм",
                "render1",
                ValidationStatus.CANNOT_VERIFY,
                0.99,
                "Точный размер по изображению не подтверждается.",
                True,
            ),
        ]

        result = AnalysisResult(
            message,
            analysis,
            ["villa_5"],
            checks,
        )
        self.assert_valid(result)

    def test_conflicting_participant_instructions(self) -> None:
        message = MessageEnvelope(
            "m7",
            "d5",
            "manager",
            Speaker.CONTACT,
            ContactRole.MANAGER,
            text="Оставляем белые фасады. Чёрные не делаем.",
            projects=[self.villa_2],
            context=MessageContext(
                None,
                ["architect_black"],
                True,
                True,
            ),
        )

        analysis = MessageAnalysis(
            message_id="m7",
            projects=[self.villa_2],
            intents=[
                IntentItem(
                    IntentType.EDITS,
                    0.98,
                    self.villa_2,
                ),
                IntentItem(
                    IntentType.REJECTION,
                    0.97,
                    self.villa_2,
                ),
            ],
            requirements=[
                Requirement(
                    "req_white",
                    "Оставить белые фасады",
                    self.villa_2,
                    "m7",
                    [],
                    [],
                    RequirementStatus.ACTIVE,
                    0.98,
                    "req_black",
                    "фасады",
                    "цвет",
                    "белый",
                    None,
                    False,
                )
            ],
            conflict=ConflictAssessment(
                False,
                True,
                ConflictLevel.MEDIUM,
                0.95,
                (
                    "Новое указание противоречит ранее "
                    "активному требованию другого участника."
                ),
                True,
            ),
            requires_human=True,
            human_reason=(
                "Нужно определить приоритет указаний участников."
            ),
            recommendation="Не менять договорённость автоматически.",
            confidence=0.95,
        )

        result = AnalysisResult(
            message,
            analysis,
            ["villa_2"],
        )
        self.assert_valid(result)

    def test_unresolved_project(self) -> None:
        unknown = ProjectReference(
            None,
            None,
            0.2,
            True,
            True,
        )

        message = MessageEnvelope(
            "m8",
            "d2",
            "manager",
            Speaker.CONTACT,
            ContactRole.MANAGER,
            text="И ещё там зеркало сделай шире.",
            projects=[unknown],
            context=MessageContext(
                None,
                ["v2_prev", "v5_prev"],
                True,
                False,
            ),
        )

        analysis = MessageAnalysis(
            message_id="m8",
            projects=[unknown],
            intents=[
                IntentItem(
                    IntentType.EDITS,
                    0.95,
                    unknown,
                )
            ],
            requirements=[
                Requirement(
                    "req_u",
                    "Сделать зеркало шире",
                    unknown,
                    "m8",
                    [],
                    [],
                    RequirementStatus.UNCLEAR,
                    0.95,
                    None,
                    "зеркало",
                    "ширина",
                    "увеличить",
                    None,
                    True,
                )
            ],
            tasks=[
                TaskItem(
                    "task_u",
                    "Сделать зеркало шире",
                    unknown,
                    source_message_id="m8",
                    status=TaskStatus.UNCLEAR,
                    requires_confirmation=True,
                )
            ],
            requires_human=True,
            human_reason="Не удалось надёжно определить проект.",
            recommendation="Уточнить проект.",
            confidence=0.80,
        )

        result = AnalysisResult(
            message,
            analysis,
            [],
        )
        self.assert_valid(result)

    def test_output_policy_and_schema_version(self) -> None:
        message = MessageEnvelope(
            "m9",
            "d6",
            "client",
            Speaker.CONTACT,
            ContactRole.CLIENT,
            text="Проверка политики вывода.",
            projects=[self.villa_2],
        )
        analysis = MessageAnalysis(
            message_id="m9",
            projects=[self.villa_2],
            confidence=0.99,
        )
        result = AnalysisResult(
            message,
            analysis,
            ["villa_2"],
        )

        self.assertEqual(result.output_policy.user_facing_language, "ru")
        self.assertTrue(result.output_policy.russian_only)
        self.assertEqual(result.schema_version, "2.4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
