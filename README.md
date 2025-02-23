# TPS_Emotion_Bot
# Emotion Classification Model

This project is an emotion classification model based on text, using LSTM and pretrained embeddings. The model is trained on textual data and predicts the emotions expressed in those texts.

## Installation

To run the project, you will need the following libraries:
```bash
pip install numpy pandas keras tensorflow nltk matplotlib
```


You also need to download some NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```


## Data

The training, validation, and test data are loaded from the files `train.txt`, `val.txt`, and `test.txt`, respectively. The files should contain text and emotion labels separated by a semicolon.

### Data Format

The data files should be in the following format:
```
Text;Emotion
```

## Text Processing

The text is processed using the following steps:
1. Removal of URLs.
2. Removal of non-alphabetic characters.
3. Conversion of text to lowercase.
4. Removal of stop words.
5. Lemmatization of words.

## Model Training

The model is a sequential network using LSTM. It is compiled with the Adam optimizer and the loss function `sparse_categorical_crossentropy`. The model is trained for 10 epochs with a callback for reducing the learning rate.

## Model Saving

After training, the model and tokenizer are saved in HDF5 and pickle formats, respectively:
```python
model.save("emotion_model.h5")
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)
```


## Prediction Example

To make a prediction on new text, you can use the following code:
```python
input_text = "I feel hurt because he cheated on me with his best friend"
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = encoder.inverse_transform([np.argmax(prediction[0])])
print(f"Predicted emotion: {predicted_label[0]}")
```
# Telegram Bot for Emotion Analysis

This project is a Telegram bot that analyzes the sentiment of text messages using a trained emotion classification model. The bot predicts the emotion expressed in the text and provides probabilities for all possible emotions.

## Installation

To run the bot, you will need the following libraries:
```bash
pip install tensorflow numpy nltk python-telegram-bot
```


You also need to download some NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Data and Model

Make sure you have the following files in the same directory as your script:
- `emotion_model.h5`: The trained model for emotion classification.
- `tokenizer.pkl`: The tokenizer used for text preprocessing.
- `encoder.pkl`: The label encoder that maps emotion labels to integers.

## How to Use

1. Clone this repository or download the files.
2. Install the required libraries using the command provided above.
3. Run the script:
```bash
python your_script_name.py
```
4. Open Telegram and find your bot using its username.
5. Start a conversation with the bot by sending the `/start` command.

## Bot Functionality

- **/start**: Sends a welcome message and instructions on how to use the bot.
- **Text Analysis**: Send any text to the bot, and it will respond with:
  - The predicted emotion.
  - Probabilities for all emotions.
  - The top 3 most likely emotions.
  - The length of the text in words.
  - The cleaned text after normalization.

## Example Interaction
```
User: I feel hurt because he cheated on me with his best friend.
Bot: 
Настроение текста: sadness

Вероятности для всех эмоций:

happiness: 0.05
sadness: 0.85
anger: 0.10
Топ-3 наиболее вероятных эмоций:

sadness: 0.85
anger: 0.10
happiness: 0.05
Длина текста: 10 слов
Очищенный текст: feel hurt cheated best friend
