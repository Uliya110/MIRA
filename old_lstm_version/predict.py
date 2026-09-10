# AI Assistant
# Intent Classifier
# predict.py
# Использование обученной LSTM модели
# Вход:
# сообщение клиента
# Выход:
# категория + уверенность модели

# ==============================
# Импорт библиотек
# ==============================
import os
import pickle
import numpy as np
# TensorFlow
from tensorflow.keras.models import load_model
# Подготовка текста
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Настройки проекта
# ==========================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "assistant_model.keras"
)
TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl"
)
LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)
# Длина сообщений.
# Она должна совпадать
# с длиной при обучении.
# Поэтому после обучения
# ее лучше сохранить.
# Сейчас берем значение,
# которое использовалось в train_classifier.py.
# ==========================================================
# Загрузка модели
# ==========================================================

print("\nЗагрузка модели...\n")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Модель не найдена. "
        "Сначала выполните обучение."
    )
model = load_model(
    MODEL_PATH
)
print(
    "✓ Модель загружена"
)
# ==========================================================
# Загрузка Tokenizer
# ==========================================================

with open(
    TOKENIZER_PATH,
    "rb"
) as file:
    tokenizer = pickle.load(file)
print(
    "✓ Tokenizer загружен"
)
# ==========================================================
# Загрузка LabelEncoder
# ==========================================================

with open(
    LABEL_ENCODER_PATH,
    "rb"
) as file:
    label_encoder = pickle.load(file)
print(
    "✓ LabelEncoder загружен"
)
# ==========================================================
# Загрузка максимальной длины сообщения
# ==========================================================

MAX_LEN_PATH = os.path.join(
    MODEL_DIR,
    "max_len.pkl"
)
with open(
    MAX_LEN_PATH,
    "rb"
) as file:
    MAX_LEN = pickle.load(file)
print(
    "✓ Максимальная длина сообщения загружена:",
    MAX_LEN
)
# ==========================================================
# Функция определения категории
# ==========================================================

def predict_intent(message):
    """
    Определяет категорию сообщения клиента.
    Вход:
        message - текст клиента
    Выход:
        категория
        вероятность
    """
    # Превращаем текст
    # в последовательность чисел
    sequence = tokenizer.texts_to_sequences(
        [message]
    )
    # Добавляем padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )
    # Получаем вероятности
    prediction = model.predict(
        padded,
        verbose=0
    )
    # Берем индекс
    # самой вероятной категории
    predicted_index = np.argmax(
        prediction
    )
    # Вероятность
    confidence = np.max(
        prediction
    )
    # Переводим число
    # обратно в название категории
    category = label_encoder.inverse_transform(

        [predicted_index]
    )[0]
    return category, confidence

# ==========================================================
# Тестирование модели
# ==========================================================

print("\n==============================")
print("AI Assistant готов")
print("==============================")
print(
    "\nВведите сообщение клиента."
)
print(
    "Для выхода напишите: exit\n"
)
while True:
    message = input(
        "Клиент: "
    )
    if message.lower() == "exit":
        print(
            "Работа завершена"
        )
        break
    category, confidence = predict_intent(
        message
    )
    print(
        "\nРезультат:"
    )
    print(
        "Категория:",
        category
    )
    print(
        "Уверенность:",
        f"{confidence:.2%}"
    )
    print(
        "-" * 40
    )