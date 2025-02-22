import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import telebot
import PIL
import asyncio
from PIL import Image
from requests import get
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import nest_asyncio

nest_asyncio.apply()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
bot = telebot.TeleBot('8143120396:AAHqHWcdpkj7ApJ-BwXLlZuIA5Cwaxvw8t4')
# Функция для очистки текста
def normalize_text(text):
    text = re.sub(r'https?://\S+|www\.\S+','', text)
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
def prepare_text(text, tokenizer, max_length):
    text = normalize_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence
def predict_emotion(model, text, tokenizer, max_length, encoder):
    padded_sequence = prepare_text(text, tokenizer, max_length)
    prediction = model.predict(padded_sequence)
    one_hot_vector = np.zeros((1, len(encoder.categories_[0])))
    one_hot_vector[0][np.argmax(prediction[0])] = 1
    predicted_label = encoder.inverse_transform(one_hot_vector)
    return predicted_label[0][0]
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Введите текст для анализа:')
async def send_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model_path = "emotion_model.h5"
    model = load_model(model_path)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    max_length = 40
    text = update.message.text
    emotion = predict_emotion(model, text, tokenizer, max_length, encoder)
    await update.message.reply_text(emotion)
async def main() -> None:
    application = ApplicationBuilder().token("8143120396:AAHqHWcdpkj7ApJ-BwXLlZuIA5Cwaxvw8t4").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_text))
    await application.run_polling()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())