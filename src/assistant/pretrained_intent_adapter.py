# -*- coding: utf-8 -*-
"""
MIRA — Pretrained Intent Adapter
================================

Рекомендуемое расположение:
    MIRA/src/assistant/pretrained_intent_adapter.py

Назначение:
- загружает локально сохранённую pretrained-модель intent:
      MIRA/models/pretrained_intent/
- реализует контракт ExternalLanguageAnalyzer;
- преобразует русский single-label baseline в IntentItem;
- НЕ заменяет rule-based/context логику AnalysisEngine.

Важно:
текущая RuBERT-модель обучалась как single-label классификатор.
Поэтому adapter возвращает один основной semantic intent.
Multi-label поведение MIRA на этом этапе получается гибридно:
    pretrained intent + rule-based secondary intents.
Позже этот adapter можно заменить настоящей multi-label моделью,
не меняя AnalysisEngine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .analysis_schema_v2 import (
    IntentItem,
    IntentType,
    MessageAnalysis,
    MessageEnvelope,
    ProjectReference,
)


TRAIN_LABEL_TO_INTENT: Dict[str, IntentType] = {
    "благодарность": IntentType.THANKS,
    "материалы": IntentType.MATERIALS,
    "оплата": IntentType.PAYMENT,
    "первичное_обращение": IntentType.PRIMARY_CONTACT,
    "правки": IntentType.EDITS,
    "рендеры": IntentType.RENDER,
    "сроки": IntentType.DEADLINE,
    "стоимость": IntentType.COST,
    "технический_вопрос": IntentType.TECHNICAL_QUESTION,
}


@dataclass
class PretrainedIntentAdapterConfig:
    model_dir: Optional[Path] = None
    device: str = "auto"
    max_length: int = 128
    local_files_only: bool = True


@dataclass
class IntentPrediction:
    train_label: str
    intent: IntentType
    confidence: float


class PretrainedIntentAdapter:
    """
    Adapter между Hugging Face sequence-classifier и MIRA MessageAnalysis.

    Совместим с контрактом:
        ExternalLanguageAnalyzer.analyze(message) -> MessageAnalysis
    """

    def __init__(
        self,
        config: Optional[PretrainedIntentAdapterConfig] = None,
    ) -> None:
        self.config = config or PretrainedIntentAdapterConfig()

        self.project_dir = self._resolve_project_dir()
        self.model_dir = (
            Path(self.config.model_dir)
            if self.config.model_dir is not None
            else self.project_dir / "models" / "pretrained_intent"
        )

        if not self.model_dir.exists():
            raise FileNotFoundError(
                "Не найдена pretrained intent-модель MIRA:\n"
                f"{self.model_dir}\n\n"
                "Сначала запустите train_pretrained_intent.py."
            )

        self.device = self._resolve_device(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=self.config.local_files_only,
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self._load_id2label()

    def analyze(self, message: MessageEnvelope) -> MessageAnalysis:
        """
        Возвращает только semantic-часть анализа.

        Tasks, requirements, deadlines, finance и другие structured fields
        остаются за hybrid AnalysisEngine.
        """
        text = (message.text or "").strip()

        if not text:
            return MessageAnalysis(
                message_id=message.message_id,
                projects=list(message.projects),
                intents=[],
                confidence=0.0,
            )

        prediction = self.predict(text)
        project = self._single_project_or_none(message.projects)

        intent_item = IntentItem(
            intent=prediction.intent,
            confidence=prediction.confidence,
            project=project,
            evidence=text,
        )

        return MessageAnalysis(
            message_id=message.message_id,
            projects=list(message.projects),
            intents=[intent_item],
            confidence=prediction.confidence,
        )

    def predict(self, text: str) -> IntentPrediction:
        predictions = self.predict_top_k(text=text, top_k=1)

        if not predictions:
            return IntentPrediction(
                train_label="",
                intent=IntentType.UNKNOWN,
                confidence=0.0,
            )

        return predictions[0]

    @torch.inference_mode()
    def predict_top_k(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[IntentPrediction]:
        """
        Диагностическая функция.

        Возвращает top-k softmax, но analyze() использует только top-1,
        потому что текущая модель обучалась как single-label.
        """
        cleaned = (text or "").strip()

        if not cleaned:
            return []

        encoded = self.tokenizer(
            cleaned,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        output = self.model(**encoded)
        probabilities = torch.softmax(output.logits, dim=-1)[0]

        safe_top_k = max(
            1,
            min(int(top_k), int(probabilities.shape[0])),
        )

        values, indices = torch.topk(probabilities, k=safe_top_k)

        result: List[IntentPrediction] = []

        for probability, index in zip(values.tolist(), indices.tolist()):
            train_label = self.id2label.get(int(index), f"LABEL_{index}")
            intent = TRAIN_LABEL_TO_INTENT.get(
                self._normalize_label(train_label),
                IntentType.UNKNOWN,
            )

            result.append(
                IntentPrediction(
                    train_label=train_label,
                    intent=intent,
                    confidence=float(probability),
                )
            )

        return result

    def _load_id2label(self) -> Dict[int, str]:
        """
        Сначала используем Hugging Face config.
        Если там остались LABEL_0... — читаем label_mapping.json,
        созданный train_pretrained_intent.py.
        """
        config_mapping = {
            int(key): str(value)
            for key, value in dict(self.model.config.id2label).items()
        }

        if self._mapping_is_semantic(config_mapping):
            self._validate_mapping(config_mapping)
            return config_mapping

        mapping_path = self.model_dir / "label_mapping.json"

        if not mapping_path.exists():
            raise ValueError(
                "В config модели нет русских intent-меток, "
                "и отсутствует label_mapping.json:\n"
                f"{mapping_path}"
            )

        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        raw_mapping = payload.get("id2label", {})
        mapping = {
            int(key): str(value)
            for key, value in raw_mapping.items()
        }

        self._validate_mapping(mapping)
        return mapping

    @staticmethod
    def _mapping_is_semantic(mapping: Dict[int, str]) -> bool:
        if not mapping:
            return False

        normalized = {
            PretrainedIntentAdapter._normalize_label(value)
            for value in mapping.values()
        }

        return bool(normalized & set(TRAIN_LABEL_TO_INTENT))

    @staticmethod
    def _validate_mapping(mapping: Dict[int, str]) -> None:
        unknown = sorted({
            label
            for label in mapping.values()
            if PretrainedIntentAdapter._normalize_label(label)
            not in TRAIN_LABEL_TO_INTENT
        })

        if unknown:
            raise ValueError(
                "Pretrained-модель содержит неизвестные MIRA labels: "
                + ", ".join(unknown)
            )

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower().replace(" ", "_")

    def _resolve_project_dir(self) -> Path:
        """
        Поддерживает:
            MIRA/src/pretrained_intent_adapter.py
            MIRA/src/assistant/pretrained_intent_adapter.py
        """
        this_file = Path(__file__).resolve()

        candidates = []
        for level in (2, 1):
            if len(this_file.parents) > level:
                candidates.append(this_file.parents[level])

        for candidate in candidates:
            if (candidate / "models").exists() and (candidate / "src").exists():
                return candidate

        for candidate in candidates:
            if (candidate / "data").exists() and (candidate / "src").exists():
                return candidate

        raise RuntimeError(
            "Не удалось определить корень проекта MIRA. "
            "Положите adapter в MIRA/src/assistant/ "
            "или передайте config.model_dir явно."
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        normalized = (requested or "auto").strip().lower()

        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if normalized == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Запрошено device='cuda', но CUDA недоступна."
                )
            return torch.device("cuda")

        if normalized == "cpu":
            return torch.device("cpu")

        raise ValueError("device должен быть: auto, cpu или cuda.")

    @staticmethod
    def _single_project_or_none(
        projects: List[ProjectReference],
    ) -> Optional[ProjectReference]:
        resolved = [project for project in projects if project.project_id]
        return resolved[0] if len(resolved) == 1 else None


if __name__ == "__main__":
    adapter = PretrainedIntentAdapter()

    samples = [
        "Нужно заменить плитку и прислать новый пререндер.",
        "Сколько будет стоить дополнительный ракурс?",
        "Когда сможете закончить?",
        "Да, получила. Спасибо!",
    ]

    print("MIRA pretrained intent adapter")
    print("Model:", adapter.model_dir)
    print("Device:", adapter.device)

    for sample in samples:
        prediction = adapter.predict(sample)
        print("\nТекст:")
        print(sample)
        print("Intent:", prediction.intent.value)
        print("Train label:", prediction.train_label)
        print("Confidence:", round(prediction.confidence, 4))
