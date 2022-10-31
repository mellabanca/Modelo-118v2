import pandas as pd 
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from datetime import datetime
# Faça um dicionário de códigos e URLs para as diferentes emoções
emo_code_url = {
    "vazio": [0, "./static/assets/emoticons/Empty.png"],
    "tristeza": [1,"./static/assets/emoticons/Sadness.png" ],
    "entusiasmo": [2, "./static/assets/emoticons/Enthusiasm.png"],
    "neutro": [3, "./static/assets/emoticons/Neutral.png"],
    "preocupacao": [4, "./static/assets/emoticons/Worry.png"],
    "surpresa": [5, "./static/assets/emoticons/Surprise.png"],
    "amor": [6, "./static/assets/emoticons/Love.png"],
    "diversao": [7, "./static/assets/emoticons/fun.png"],
    "odio": [8, "./static/assets/emoticons/hate.png"],
    "felicidade": [9, "./static/assets/emoticons/happiness.png"],
    "tedio": [10, "./static/assets/emoticons/boredom.png"],
    "alivio": [11, "./static/assets/emoticons/relief.png"],
    "raiva": [12, "./static/assets/emoticons/anger.png"],
}


train_data = pd.read_csv("./static/assets/data_files/tweet_emotions.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

model = load_model("./static/assets/model_file/Tweet_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

def predict(text):
    predicted_emotion_img_url=""
    predicted_emotion=""

    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)        
            
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url

#Exiba a entrada
def show_entry():
    day_entry_list = pd.read_csv("./static/assets/data_files/data_entry.csv")

    day_entry_list = day_entry_list.iloc[::-1]
    
    date1 = (day_entry_list['date'].values[0])
    date2 =(day_entry_list['date'].values[1])
    date3 = (day_entry_list['date'].values[2])

    entry1 = day_entry_list['text'].values[0]
    entry2 = day_entry_list['text'].values[1]
    entry3 = day_entry_list['text'].values[2]

    emotion1 = day_entry_list["emotion"].values[0]
    emotion2 = day_entry_list["emotion"].values[1]
    emotion3 = day_entry_list["emotion"].values[2]

    emotion_url_1=""
    emotion_url_2=""
    emotion_url_3=""

    for key, value in emo_code_url.items():
        if key==emotion1:
            emotion_url_1 = value[1]
        if key==emotion2:
            emotion_url_2 = value[1]
        if key==emotion3:
            emotion_url_3 = value[1]

    return [
        {
            "date": date1,
            "entry": entry1,
            "emotion": emotion1,
            "emotion_url": emotion_url_1
        },
        {
            "date": date2,
            "entry": entry2,
            "emotion": emotion2,
            "emotion_url": emotion_url_2
        },
        {
            "date": date3,
            "entry": entry3,
            "emotion": emotion3,
            "emotion_url": emotion_url_3
        }
    ]
