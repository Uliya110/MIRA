"""
MIRA — Reminder Service
=======================

Сервис доставки напоминаний.

Этот модуль отвечает за последний участок цепочки:

    сообщение / договорённость / срок
              ↓
       AnalysisResult
              ↓
     ProjectStateEngine
              ↓
        ReminderItem
              ↓
       ReminderService
              ↓
    Telegram / интерфейс MIRA

ВАЖНО:
ProjectStateEngine создаёт и хранит напоминания.
ReminderService проверяет, наступило ли время уведомления,
и передаёт готовый русский текст в канал доставки.

Этот файл НЕ:
- анализирует сообщения;
- не извлекает даты из текста;
- не решает, относится ли сообщение к проекту;
- не генерирует смысл напоминания через LLM.

Он работает только с уже созданными ReminderItem.

Для MVP используется in-memory хранилище ProjectStateEngine.
Позже можно заменить его на PostgreSQL/Redis без изменения
основной логики ReminderService.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Protocol

from .analysis_schema_v2 import (
    ReminderItem,
    ReminderStatus,
)


# ---------------------------------------------------------------------
# КОНТРАКТ КАНАЛА ДОСТАВКИ
# ---------------------------------------------------------------------

class ReminderSender(Protocol):
    """
    Любой канал доставки напоминаний должен реализовать send().

    В будущем это может быть:
    - TelegramReminderSender
    - DesktopNotificationSender
    - EmailReminderSender
    - WebSocketReminderSender

    Благодаря этому ReminderService не зависит от конкретного интерфейса.
    """

    def send(self, reminder: ReminderItem) -> bool:
        """
        Отправляет одно напоминание.

        Возвращает:
            True  — если доставка принята/успешна;
            False — если отправить не удалось.
        """
        ...


class ConsoleReminderSender:
    """
    Простейший sender для разработки и тестов.

    Вместо Telegram просто печатает текст в консоль.
    """

    def send(self, reminder: ReminderItem) -> bool:
        print(
            f"[MIRA reminder] {reminder.user_message_ru}"
        )
        return True


class CallbackReminderSender:
    """
    Универсальный адаптер.

    Позволяет передать обычную Python-функцию вместо отдельного класса.

    Пример:

        def send_to_telegram(reminder):
            ...
            return True

        sender = CallbackReminderSender(send_to_telegram)
    """

    def __init__(
        self,
        callback: Callable[[ReminderItem], bool],
    ) -> None:
        self.callback = callback

    def send(self, reminder: ReminderItem) -> bool:
        return bool(self.callback(reminder))


# ---------------------------------------------------------------------
# РЕЗУЛЬТАТ ОДНОГО ЦИКЛА ПРОВЕРКИ
# ---------------------------------------------------------------------

@dataclass
class ReminderRunResult:
    """
    Итог одного запуска ReminderService.run_once().

    checked:
        сколько pending reminders было просмотрено.

    due:
        сколько из них уже должны были сработать.

    sent:
        сколько успешно отправлено.

    failed:
        сколько отправить не удалось.
    """
    checked: int = 0
    due: int = 0
    sent: int = 0
    failed: int = 0


# ---------------------------------------------------------------------
# АДАПТЕР К ТЕКУЩЕМУ PROJECT STATE ENGINE
# ---------------------------------------------------------------------

class ProjectStateReminderStore:
    """
    Адаптер к текущему in-memory ProjectStateEngine.

    Сейчас ProjectStateEngine v3 умеет:
        get_reminders(project_id, pending_only=True)

    Но пока ещё не имеет отдельного публичного метода:
        mark_reminder_sent(...)

    Поэтому этот адаптер инкапсулирует временную работу
    с engine._reminders.

    Это намеренный MVP-компромисс:
    ReminderService сам НЕ обращается к приватным полям движка —
    это делает только этот маленький адаптер.

    Позже, когда появится Repository/DB слой, этот класс можно заменить.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    def get_pending(
        self,
        project_id: Optional[str] = None,
    ) -> List[ReminderItem]:
        """
        Возвращает pending reminders.

        Если project_id не указан — собирает напоминания
        по всем проектам движка.
        """
        if project_id is not None:
            return self.engine.get_reminders(
                project_id,
                pending_only=True,
            )

        result: List[ReminderItem] = []

        # get_all_states() — публичный метод движка.
        for pid in self.engine.get_all_states().keys():
            result.extend(
                self.engine.get_reminders(
                    pid,
                    pending_only=True,
                )
            )

        return result

    def mark_sent(self, reminder_id: str) -> bool:
        """
        Переводит reminder в SENT.

        Возвращает False, если напоминание не найдено.
        """
        return self._set_status(
            reminder_id,
            ReminderStatus.SENT,
        )

    def mark_cancelled(self, reminder_id: str) -> bool:
        """
        Позволяет сервису отменить конкретное напоминание.
        """
        return self._set_status(
            reminder_id,
            ReminderStatus.CANCELLED,
        )

    def _set_status(
        self,
        reminder_id: str,
        status: ReminderStatus,
    ) -> bool:
        """
        Временный bridge к in-memory storage ProjectStateEngine.

        Единственное место в reminder-модуле, где допускается
        доступ к engine._reminders.
        """
        reminders_by_project = getattr(
            self.engine,
            "_reminders",
            None,
        )

        if reminders_by_project is None:
            raise RuntimeError(
                "ProjectStateEngine не предоставляет "
                "in-memory reminder storage."
            )

        for reminders in reminders_by_project.values():
            for reminder in reminders:
                if reminder.reminder_id == reminder_id:
                    reminder.status = status
                    return True

        return False


# ---------------------------------------------------------------------
# REMINDER SERVICE
# ---------------------------------------------------------------------

class ReminderService:
    """
    Проверяет pending reminders и отправляет те,
    у которых наступил trigger_at_iso.

    Пример использования:

        service = ReminderService(
            store=ProjectStateReminderStore(engine),
            sender=TelegramReminderSender(...)
        )

        service.run_once()

    В реальной MIRA run_once() должен запускаться scheduler'ом
    регулярно, например раз в минуту.

    Сам ReminderService намеренно не создаёт бесконечный цикл —
    так его проще тестировать и безопаснее встраивать в приложение.
    """

    def __init__(
        self,
        store: ProjectStateReminderStore,
        sender: ReminderSender,
    ) -> None:
        self.store = store
        self.sender = sender

    def run_once(
        self,
        now: Optional[datetime] = None,
        project_id: Optional[str] = None,
    ) -> ReminderRunResult:
        """
        Выполняет одну проверку.

        now:
            можно передать вручную в тестах.
            Если не передан — используется текущее локальное время.

        project_id:
            если указан, проверяется только один проект.
        """
        current_time = now or datetime.now()

        reminders = self.store.get_pending(
            project_id=project_id,
        )

        result = ReminderRunResult(
            checked=len(reminders),
        )

        for reminder in reminders:
            trigger_at = self._parse_iso(
                reminder.trigger_at_iso,
            )

            # Напоминание ещё не наступило.
            if trigger_at > current_time:
                continue

            result.due += 1

            # ---------------------------------------------------------
            # ЯЗЫКОВАЯ ЗАЩИТА
            # ---------------------------------------------------------
            # В schema уже зафиксировано russian_only=True.
            # Здесь делаем дополнительную проверку:
            # пустой пользовательский текст не отправляем.
            #
            # Полноценное определение языка здесь не нужно:
            # ReminderService не должен заниматься NLP.
            # ---------------------------------------------------------
            if not reminder.user_message_ru.strip():
                result.failed += 1
                continue

            try:
                delivered = self.sender.send(reminder)
            except Exception as exc:
                # В production здесь будет logging.
                print(
                    f"Ошибка отправки reminder "
                    f"{reminder.reminder_id}: {exc}"
                )
                delivered = False

            if delivered:
                self.store.mark_sent(
                    reminder.reminder_id
                )
                result.sent += 1
            else:
                # Оставляем PENDING.
                # Следующий запуск сможет повторить попытку.
                result.failed += 1

        return result

    def get_due_reminders(
        self,
        now: Optional[datetime] = None,
        project_id: Optional[str] = None,
    ) -> List[ReminderItem]:
        """
        Возвращает reminders, которые уже должны быть отправлены,
        но ничего не отправляет.

        Полезно:
        - для интерфейса;
        - для отладки;
        - для страницы "Просроченные уведомления".
        """
        current_time = now or datetime.now()

        result: List[ReminderItem] = []

        for reminder in self.store.get_pending(project_id):
            trigger_at = self._parse_iso(
                reminder.trigger_at_iso
            )

            if trigger_at <= current_time:
                result.append(reminder)

        return result

    def get_next_reminder(
        self,
        project_id: Optional[str] = None,
    ) -> Optional[ReminderItem]:
        """
        Возвращает ближайшее ожидающее напоминание.

        Ничего не отправляет.
        """
        reminders = self.store.get_pending(project_id)

        if not reminders:
            return None

        return min(
            reminders,
            key=lambda item: self._parse_iso(
                item.trigger_at_iso
            ),
        )

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        """
        Преобразует ISO-строку в datetime.

        В schema даты нормализуются заранее.
        Поэтому ReminderService не пытается понимать
        фразы "завтра вечером" — это работа Analysis Engine.
        """
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Некорректный trigger_at_iso: {value}"
            ) from exc


# ---------------------------------------------------------------------
# ПРИМЕР ДЛЯ ЛОКАЛЬНОГО ЗАПУСКА
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Мини-демонстрация без Telegram.

    Создаём fake engine с одним напоминанием,
    затем запускаем сервис.
    """

    from .analysis_schema_v2 import (
        ReminderKind,
    )

    class FakeEngine:
        """
        Минимальная имитация ProjectStateEngine
        только для демонстрации этого файла.
        """

        def __init__(self):
            self._reminders = {
                "villa_2": [
                    ReminderItem(
                        reminder_id="rem_demo_001",
                        project_id="villa_2",
                        kind=ReminderKind.TASK_DEADLINE,
                        trigger_at_iso="2026-08-30T17:00:00",
                        due_at_iso="2026-08-30T18:00:00",
                        source_message_id="msg_001",
                        source_entity_id="task_final",
                        user_message_ru=(
                            "Напоминание по проекту «Villa 2»: "
                            "срок задачи «Отправить финальный рендер» "
                            "сегодня в 18:00."
                        ),
                        status=ReminderStatus.PENDING,
                        notify_before_minutes=60,
                    )
                ]
            }

        def get_all_states(self):
            # Для ReminderService важны только ID проектов.
            return {"villa_2": object()}

        def get_reminders(
            self,
            project_id,
            pending_only=True,
        ):
            reminders = list(
                self._reminders.get(project_id, [])
            )

            if pending_only:
                reminders = [
                    x for x in reminders
                    if x.status == ReminderStatus.PENDING
                ]

            # В настоящем engine возвращается deepcopy.
            return reminders

    fake_engine = FakeEngine()

    service = ReminderService(
        store=ProjectStateReminderStore(fake_engine),
        sender=ConsoleReminderSender(),
    )

    run_result = service.run_once(
        now=datetime.fromisoformat(
            "2026-08-30T17:00:00"
        )
    )

    print()
    print("Проверено:", run_result.checked)
    print("Наступило:", run_result.due)
    print("Отправлено:", run_result.sent)
    print("Ошибок:", run_result.failed)
