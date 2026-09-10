"""
MIRA — Analysis Schema v2.4
===========================

Единый контракт данных для анализа двусторонней рабочей коммуникации.

Ключевые принципы:
- анализируются входящие и исходящие сообщения;
- короткие ответы пользователя трактуются только в контексте;
- один контакт может участвовать в нескольких проектах;
- одно сообщение может относиться сразу к нескольким проектам;
- задачи/требования имеют собственную привязку к проекту;
- задачи могут быть явно связаны с требованиями через related_requirement_ids;
- учитываются вложения и ссылки;
- фиксируются требования, решения, договорённости, события и финансы;
- конфликт оценивается и на уровне сообщения, и на уровне диалога;
- визуальная проверка пререндеров носит рекомендательный характер;
- 3ds Max-сцена не является объектом автоматической проверки;
- ВСЕ пользовательские ответы MIRA формируются только на русском языке.

Внутренние имена полей, enum-значения и технические идентификаторы
могут оставаться на английском.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------

class Speaker(str, Enum):
    USER = "user"
    CONTACT = "contact"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ContactRole(str, Enum):
    CLIENT = "client"
    ARCHITECT = "architect"
    MANAGER = "manager"
    ACCOUNTANT = "accountant"
    CONTRACTOR = "contractor"
    COLLEAGUE = "colleague"
    OTHER = "other"
    UNKNOWN = "unknown"


class MessageType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    LINK = "link"
    MIXED = "mixed"


class AttachmentType(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    DRAWING = "drawing"
    SPREADSHEET = "spreadsheet"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    OTHER = "other"


class LinkType(str, Enum):
    REFERENCE = "reference"
    MODEL = "model"
    PRODUCT = "product"
    CLOUD_FILE = "cloud_file"
    CLOUD_FOLDER = "cloud_folder"
    VIDEO = "video"
    WEBSITE = "website"
    DOCUMENT = "document"
    OTHER = "other"
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    PRIMARY_CONTACT = "primary_contact"
    EDITS = "edits"
    RENDER = "render"
    MATERIALS = "materials"
    TECHNICAL_QUESTION = "technical_question"
    COST = "cost"
    PAYMENT = "payment"
    DEADLINE = "deadline"
    THANKS = "thanks"

    NEW_TASK = "new_task"
    TASK_STATUS = "task_status"
    PRIORITY_CHANGE = "priority_change"
    APPROVAL = "approval"
    REJECTION = "rejection"
    CLARIFICATION = "clarification"
    AVAILABILITY = "availability"
    DOCUMENTS = "documents"
    LOGISTICS = "logistics"
    FILE_TRANSFER = "file_transfer"
    RESULT_DELIVERY = "result_delivery"

    OTHER = "other"
    UNKNOWN = "unknown"


class PriorityLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ConflictLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RequirementStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REPLACED = "replaced"
    UNCLEAR = "unclear"


class TaskStatus(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    WAITING_FOR_CONTACT = "waiting_for_contact"
    WAITING_FOR_USER = "waiting_for_user"
    UNCLEAR = "unclear"


class DecisionStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    UNCLEAR = "unclear"


class MessageEventType(str, Enum):
    PREVIEW_SENT = "preview_sent"
    FINAL_RENDER_SENT = "final_render_sent"
    FILE_SENT = "file_sent"
    FILE_RECEIVED = "file_received"
    LINK_SENT = "link_sent"
    LINK_RECEIVED = "link_received"
    PAYMENT_RECEIVED = "payment_received"
    PAYMENT_SENT = "payment_sent"
    APPROVAL_RECEIVED = "approval_received"
    APPROVAL_SENT = "approval_sent"
    TASK_ACCEPTED = "task_accepted"
    TASK_PAUSED = "task_paused"
    TASK_CANCELLED = "task_cancelled"
    DEADLINE_CONFIRMED = "deadline_confirmed"
    OTHER = "other"


class ReminderKind(str, Enum):
    TASK_DEADLINE = "task_deadline"
    AGREEMENT_DEADLINE = "agreement_deadline"
    SCHEDULED_EVENT = "scheduled_event"
    FOLLOW_UP = "follow_up"
    OTHER = "other"


class ReminderStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class ValidationStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    MATCH = "match"
    POSSIBLE_MISMATCH = "possible_mismatch"
    MISMATCH = "mismatch"
    CANNOT_VERIFY = "cannot_verify"
    NEEDS_HUMAN = "needs_human"


# ---------------------------------------------------------------------
# SOURCES / CONTEXT
# ---------------------------------------------------------------------

@dataclass
class ProjectReference:
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    confidence: float = 0.0
    inferred_from_context: bool = False
    requires_confirmation: bool = False


@dataclass
class Attachment:
    attachment_id: str
    attachment_type: AttachmentType
    file_name: Optional[str] = None
    mime_type: Optional[str] = None

    description: Optional[str] = None
    extracted_text: Optional[str] = None

    content_analyzed: bool = False
    analysis_confidence: Optional[float] = None

    project: Optional[ProjectReference] = None


@dataclass
class Link:
    """
    Ссылка остаётся частью истории проекта даже тогда, когда MIRA
    не может открыть или проанализировать её содержимое.
    """
    link_id: str
    url: str
    link_type: LinkType = LinkType.UNKNOWN

    description: Optional[str] = None
    purpose: Optional[str] = None

    project: Optional[ProjectReference] = None
    source_message_id: Optional[str] = None

    content_analyzed: bool = False
    analysis_confidence: Optional[float] = None

    access_required: bool = False
    access_error: Optional[str] = None


@dataclass
class MessageContext:
    reply_to_message_id: Optional[str] = None
    previous_message_ids: List[str] = field(default_factory=list)
    context_required: bool = False
    context_resolved: bool = False


@dataclass
class MessageEnvelope:
    message_id: str
    dialog_id: str
    contact_id: str

    speaker: Speaker
    contact_role: ContactRole = ContactRole.UNKNOWN

    timestamp: Optional[datetime] = None
    message_type: MessageType = MessageType.TEXT

    text: str = ""
    attachments: List[Attachment] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)

    projects: List[ProjectReference] = field(default_factory=list)
    context: MessageContext = field(default_factory=MessageContext)


# ---------------------------------------------------------------------
# SEMANTIC ANALYSIS
# ---------------------------------------------------------------------

@dataclass
class IntentItem:
    intent: IntentType
    confidence: float
    project: Optional[ProjectReference] = None
    evidence: Optional[str] = None


@dataclass
class TaskItem:
    task_id: Optional[str]
    description: str

    project: Optional[ProjectReference] = None
    priority: PriorityLevel = PriorityLevel.NORMAL

    deadline_text: Optional[str] = None
    deadline_iso: Optional[str] = None
    deadline_confirmed_by_user: bool = False

    source_message_id: Optional[str] = None
    source_attachment_ids: List[str] = field(default_factory=list)
    source_link_ids: List[str] = field(default_factory=list)

    # Явная связь задачи с требованиями проекта.
    #
    # Это нужно, чтобы ProjectStateEngine понимал, что при замене,
    # отмене или завершении требования может потребоваться обновить
    # связанную рабочую задачу.
    #
    # Пример:
    # requirement "шкаф 350 мм" -> task "настроить шкаф 350 мм"
    # новое requirement "шкаф 250 мм" supersedes старое
    # движок сможет найти связанную задачу и пометить её устаревшей.
    related_requirement_ids: List[str] = field(default_factory=list)

    status: TaskStatus = TaskStatus.NEW
    requires_confirmation: bool = False


@dataclass
class FinancialIssue:
    detected: bool = False
    topic: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None

    # asked / proposed / confirmed / paid / overdue / unknown
    status: Optional[str] = None

    project: Optional[ProjectReference] = None
    requires_human: bool = False


@dataclass
class ConflictAssessment:
    message_conflict: bool = False
    dialog_conflict: bool = False

    level: ConflictLevel = ConflictLevel.NONE
    confidence: float = 0.0

    reason: Optional[str] = None
    requires_human: bool = False


@dataclass
class Requirement:
    requirement_id: Optional[str]
    description: str

    project: Optional[ProjectReference] = None

    source_message_id: Optional[str] = None
    source_attachment_ids: List[str] = field(default_factory=list)
    source_link_ids: List[str] = field(default_factory=list)

    status: RequirementStatus = RequirementStatus.NEW
    confidence: float = 0.0

    # Если новое требование заменяет прежнее.
    supersedes_requirement_id: Optional[str] = None

    entity: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None

    requires_human_confirmation: bool = False


@dataclass
class Decision:
    decision_id: Optional[str]
    description: str

    project: Optional[ProjectReference] = None
    related_requirement_ids: List[str] = field(default_factory=list)

    source_message_id: Optional[str] = None
    source_attachment_ids: List[str] = field(default_factory=list)
    source_link_ids: List[str] = field(default_factory=list)

    status: DecisionStatus = DecisionStatus.PROPOSED
    confirmed_by_user: bool = False

    confidence: float = 0.0
    inferred_from_context: bool = False


@dataclass
class Agreement:
    agreement_id: Optional[str]
    description: str

    project: Optional[ProjectReference] = None
    related_requirement_ids: List[str] = field(default_factory=list)
    related_decision_ids: List[str] = field(default_factory=list)

    active: bool = True

    # Срок/дата договорённости.
    # deadline_text хранит исходную формулировку ("до пятницы"),
    # deadline_iso — нормализованную дату/время, если её удалось определить.
    deadline_text: Optional[str] = None
    deadline_iso: Optional[str] = None
    reminder_required: bool = False

    source_message_ids: List[str] = field(default_factory=list)
    source_attachment_ids: List[str] = field(default_factory=list)
    source_link_ids: List[str] = field(default_factory=list)

    confirmed_by: List[str] = field(default_factory=list)


@dataclass
class MessageEvent:
    event_id: Optional[str]
    event_type: MessageEventType

    project: Optional[ProjectReference] = None
    source_message_id: Optional[str] = None
    source_attachment_ids: List[str] = field(default_factory=list)
    source_link_ids: List[str] = field(default_factory=list)

    description: Optional[str] = None

    # Если событие относится к будущей конкретной дате/времени.
    scheduled_for_text: Optional[str] = None
    scheduled_for_iso: Optional[str] = None
    reminder_required: bool = False

    confidence: float = 0.0
    requires_human_confirmation: bool = False


@dataclass
class VisualRequirementCheck:
    """
    Визуальная проверка носит рекомендательный характер.
    Точные физические параметры нельзя подтверждать по изображению,
    если для этого нет надёжных данных.
    """
    requirement_id: Optional[str]
    description: str

    source_render_attachment_id: Optional[str] = None

    status: ValidationStatus = ValidationStatus.NOT_CHECKED
    confidence: float = 0.0

    explanation: Optional[str] = None
    requires_human_confirmation: bool = True


@dataclass
class MessageAnalysis:
    message_id: str

    projects: List[ProjectReference] = field(default_factory=list)
    intents: List[IntentItem] = field(default_factory=list)

    tasks: List[TaskItem] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    agreements: List[Agreement] = field(default_factory=list)

    financial_issues: List[FinancialIssue] = field(default_factory=list)
    conflict: ConflictAssessment = field(default_factory=ConflictAssessment)
    events: List[MessageEvent] = field(default_factory=list)

    urgency: PriorityLevel = PriorityLevel.NORMAL

    requires_human: bool = False
    human_reason: Optional[str] = None

    recommendation: Optional[str] = None
    proposed_response: Optional[str] = None

    confidence: float = 0.0


@dataclass
class ReminderItem:
    """
    Напоминание, сформированное из срока задачи, договорённости
    или будущего события.

    user_message_ru — готовый пользовательский текст напоминания.
    Он обязан быть на русском языке.
    """
    reminder_id: str
    project_id: str
    kind: ReminderKind

    trigger_at_iso: str
    due_at_iso: Optional[str] = None

    source_message_id: Optional[str] = None
    source_entity_id: Optional[str] = None

    user_message_ru: str = ""
    status: ReminderStatus = ReminderStatus.PENDING

    # За сколько минут до срока напомнить.
    # Для MVP используем одно значение; позже можно хранить несколько.
    notify_before_minutes: int = 60


# ---------------------------------------------------------------------
# CURRENT PROJECT STATE
# ---------------------------------------------------------------------

@dataclass
class ProjectState:
    project_id: str
    project_name: str

    contact_ids: List[str] = field(default_factory=list)

    active_requirements: List[Requirement] = field(default_factory=list)
    active_agreements: List[Agreement] = field(default_factory=list)
    open_tasks: List[TaskItem] = field(default_factory=list)

    open_questions: List[str] = field(default_factory=list)

    last_updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------
# USER-FACING LANGUAGE POLICY
# ---------------------------------------------------------------------

USER_FACING_LANGUAGE = "ru"


@dataclass
class OutputPolicy:
    """
    Обязательная политика интерфейса MIRA.

    Все ответы, рекомендации, предупреждения, сводки, пояснения,
    вопросы на подтверждение и предлагаемые ответы, которые видит
    пользователь, должны формироваться ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.

    Внутренние поля и технические идентификаторы могут быть английскими.
    """
    user_facing_language: str = USER_FACING_LANGUAGE
    russian_only: bool = True


# ---------------------------------------------------------------------
# FULL RESULT
# ---------------------------------------------------------------------

@dataclass
class AnalysisResult:
    message: MessageEnvelope
    analysis: MessageAnalysis

    affected_project_ids: List[str] = field(default_factory=list)
    visual_checks: List[VisualRequirementCheck] = field(default_factory=list)

    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    schema_version: str = "2.4"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------------------

def validate_analysis_result(result: AnalysisResult) -> List[str]:
    """
    Проверяет базовую целостность данных.
    Это не проверка качества ML/LLM-анализа.
    """
    errors: List[str] = []

    if not result.message.message_id:
        errors.append("message_id is required")

    if not result.message.dialog_id:
        errors.append("dialog_id is required")

    if not result.message.contact_id:
        errors.append("contact_id is required")

    if result.analysis.message_id != result.message.message_id:
        errors.append("analysis.message_id must match message.message_id")

    for intent in result.analysis.intents:
        if not 0.0 <= intent.confidence <= 1.0:
            errors.append(f"intent confidence out of range: {intent.intent}")

    if not 0.0 <= result.analysis.confidence <= 1.0:
        errors.append("analysis confidence must be between 0 and 1")

    for ref in result.analysis.projects:
        if not 0.0 <= ref.confidence <= 1.0:
            errors.append("project confidence must be between 0 and 1")

    if result.output_policy.user_facing_language != "ru":
        errors.append("MIRA user-facing language must be Russian ('ru')")

    if not result.output_policy.russian_only:
        errors.append("MIRA russian_only policy must be enabled")

    return errors
