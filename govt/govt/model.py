import json
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# Load intents JSON
with open("sample.json", encoding="utf-8") as myfile:
    data = json.load(myfile)

try:
    with open("assets/input_data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("assets/input_data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# ✅ Build Keras model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(len(output[0]), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# Try loading trained model, else train
try:
    model = load_model("assets/chatbot_model.keras")
    print("Model loaded from file.")
except:
    print("Training new model...")
    model.fit(training, output, epochs=200, batch_size=8, verbose=1)
    model.save("assets/chatbot_model.keras")

# Convert sentence into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Chat loop
def start_chat():
    print("\n\nBot is ready to talk to you. (type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() in ["quit", "exit"]:
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] < 0.8 or len(inp) < 2:
            print("Bot: Sorry, I didn't get you. Please try again.\n")
        else:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            print("Bot: " + random.choice(responses) + "\n")


start_chat()
