import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import nest_asyncio

nest_asyncio.apply()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def normalize_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    lemmatizer = WordNetLemmatizer()
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    return text


# Загрузка модели
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# Подготовка текста
def prepare_text(text, tokenizer, max_length):
    text = normalize_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence

def predict_emotion_with_probability(model, text, tokenizer, max_length, encoder):
    padded_sequence = prepare_text(text, tokenizer, max_length)
    prediction = model.predict(padded_sequence)
    probabilities = prediction[0]
    predicted_index = np.argmax(probabilities)

    predicted_label = encoder.inverse_transform([predicted_index])[0]
    return predicted_label, probabilities


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Привет! Я бот для анализа настроения текста. Отправь мне текст, и я скажу, какое у него настроение.')


async def send_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model_path = "emotion_model.h5"
    model = load_model(model_path)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    max_length = 40
    text = update.message.text

    emotion, probabilities = predict_emotion_with_probability(model, text, tokenizer, max_length, encoder)

    response = f"Настроение текста: **{emotion}**\n\n"

    response += "Вероятности для всех эмоций:\n"
    for i, label in enumerate(encoder.classes_):
        response += f"- {label}: {probabilities[i]:.2f}\n"

    # Топ-3 наиболее вероятных эмоций
    top_indices = np.argsort(probabilities)[-3:][::-1]
    response += "\nТоп-3 наиболее вероятных эмоций:\n"
    for idx in top_indices:
        response += f"- {encoder.classes_[idx]}: {probabilities[idx]:.2f}\n"

    # Дополнительная информация
    response += f"\nДлина текста: {len(text.split())} слов\n"
    response += f"Очищенный текст: {normalize_text(text)}\n"

    # Отправка ответа
    await update.message.reply_text(response)


# Основная функция
async def main() -> None:
    application = ApplicationBuilder().token("8143120396:AAHqHWcdpkj7ApJ-BwXLlZuIA5Cwaxvw8t4").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_text))
    await application.run_polling()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
