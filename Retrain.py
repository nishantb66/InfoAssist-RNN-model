# Import necessary libraries
import os
import json
import keras
import numpy as np
import tensorflow

# from tensorflow.keras.models import Sequential
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout

# from keras.optimizers import SGD
import nltk
import streamlit as st

# nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()


# Initialize variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ["!", "?", ",", "."]


# Function to parse the .txt file
# Function to parse the .txt file
def parse_txt_data(file_path):
    intents = []
    current_tag = ""
    patterns = []
    responses = []
    with open(file_path, "r") as file:
        for line in file:
            # Each line contains a question or an answer, starting with 'Q: ' or 'A: '
            if line.startswith("Q: "):
                # This is a question, so we start a new intent
                current_tag = f"tag_{len(intents)}"
                patterns = [line[3:].strip()]  # Remove 'Q: ' from the start
            elif line.startswith("A: "):
                # This is an answer, so we add it to the current intent
                responses = [line[3:].strip()]  # Remove 'A: ' from the start
                intents.append(
                    {
                        "tag": current_tag,
                        "patterns": patterns,
                        "responses": responses,
                    }
                )
    return {"intents": intents}


# Load and parse your dataset
data = parse_txt_data("dialogs.txt")

if "intents" in data and data["intents"]:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents
            documents.append((word_list, intent["tag"]))
            # Add to classes list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Create training data
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    # Ensure training is a two-dimensional array
    training = np.array(training, dtype=object)

    # Check if training is not a 2D array
    if len(training.shape) == 1:
        # Reshape training to be a 2D array if it's currently 1D
        training = np.array([training.tolist()])

    # Now you can safely access train_x and train_y
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Check if the model already exists
    # if os.path.exists("chatbot_model.h5"):
    #     # Load the existing model
    #     model = load_model("chatbot_model.h5")
    # else:
    # Create the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    # model.add(Dense(16, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    # Compile the model
    # optimizer to 'adam' and learning rate to 0.001
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
    )

    callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    # Train the model
    model.fit(
        np.array(train_x),
        np.array(train_y),
        epochs=2000,
        batch_size=10,
        verbose=1,
        callbacks=callback,
    )

    # Save words and classes
    with open("words.json", "w") as f:
        json.dump(words, f)
    with open("classes.json", "w") as f:
        json.dump(classes, f)
    with open("intents.json", "w") as f:
        json.dump(data, f)

    # Save the model
    model.save("chatbot_model.h5", overwrite=True)

    