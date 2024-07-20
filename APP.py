# from flask import Flask, render_template, request, jsonify
# import os
# import json
# import keras
# import numpy as np
# import nltk
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from textblob import TextBlob


# # from transformers import pipeline

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# vectorizer = TfidfVectorizer()
# words = []
# classes = []
# documents = []
# ignore_letters = ["!", "?", ",", "."]

# # bert_qa = pipeline(
# #     "question-answering", model="distilbert-base-uncased-distilled-squad"
# # )


# # Load pre-trained model and data
# model = load_model("chatbot_model.h5")
# with open("words.json") as f:
#     words = json.load(f)
# with open("classes.json") as f:
#     classes = json.load(f)
# with open("intents.json") as f:
#     intents_data = json.load(f)


# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words


# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print(f"found in bag: {w}")
#     return np.array(bag)


# def predict_class(sentence, model):
#     p = bow(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list


# def get_response(intents_list, intents_json):
#     if intents_list:
#         tag = intents_list[0]["intent"]
#         list_of_intents = intents_json["intents"]
#         for i in list_of_intents:
#             if i["tag"] == tag:
#                 result = i["responses"][0]
#                 break
#     else:
#         result = "I'm sorry, I don't have this information. You can take the question assistance."
#     return result


# def get_alternative_questions(user_input, data):
#     all_questions = [
#         pattern for intent in data["intents"] for pattern in intent["patterns"]
#     ]
#     tfidf_matrix = vectorizer.fit_transform([user_input] + all_questions)
#     cosine_similarities = cosine_similarity(
#         tfidf_matrix[0:1], tfidf_matrix[1:]
#     ).flatten()
#     top_question_indices = cosine_similarities.argsort()[:-6:-1]
#     top_questions = [all_questions[i] for i in top_question_indices]
#     return top_questions


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/login")
# def login():
#     return render_template("login.html")


# @app.route("/admin")
# def admin():
#     return render_template("admin.html")

# @app.route("/manual")
# def manual():
#     return render_template("user_manual.html")


# def analyze_sentiment(text):
#     analysis = TextBlob(text)
#     return analysis.sentiment.polarity, analysis.sentiment.subjectivity


# @app.route("/get", methods=["POST"])
# def chatbot_response():
#     data = request.get_json()
#     msg = data["msg"]
#     feedback = data.get("feedback", "").lower()
#     if feedback == "no":
#         alternatives = get_alternative_questions(msg, intents_data)
#         return jsonify(
#             {
#                 "response": "",
#                 "alternatives": alternatives,
#             }
#         )
#     elif feedback.isdigit():
#         index = int(feedback) - 1
#         alternative_questions = get_alternative_questions(msg, intents_data)
#         selected_question = alternative_questions[index]
#         ints = predict_class(selected_question, model)
#         res = get_response(ints, intents_data)
#         return jsonify({"response": res})
#     else:
#         ints = predict_class(msg, model)
#         res = get_response(ints, intents_data)
#         return jsonify({"response": res, "alternatives": []})


# @app.route("/add", methods=["POST"])
# def add_to_dataset():
#     data = request.get_json()
#     question = data["question"]
#     answer = data["answer"]

#     with open("dialogs.txt", "a") as f:
#         f.write(f"Q: {question}\nA: {answer}\n\n")

#     return jsonify({"status": "Question and answer added successfully."})


# @app.route("/retrain", methods=["POST"])
# def retrain_model():
#     data = parse_txt_data("dialogs.txt")
#     if "intents" in data and data["intents"]:
#         global words, classes, documents, model
#         words = []
#         classes = []
#         documents = []
#         for intent in data["intents"]:
#             for pattern in intent["patterns"]:
#                 word_list = nltk.word_tokenize(pattern)
#                 words.extend(word_list)
#                 documents.append((word_list, intent["tag"]))
#                 if intent["tag"] not in classes:
#                     classes.append(intent["tag"])

#         words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
#         words = sorted(list(set(words)))
#         classes = sorted(list(set(classes)))

#         training = []
#         output_empty = [0] * len(classes)

#         for document in documents:
#             bag = []
#             word_patterns = document[0]
#             word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#             for word in words:
#                 bag.append(1) if word in word_patterns else bag.append(0)

#             output_row = list(output_empty)
#             output_row[classes.index(document[1])] = 1
#             training.append([bag, output_row])

#         training = np.array(training, dtype=object)

#         if len(training.shape) == 1:
#             training = np.array([training.tolist()])

#         train_x = list(training[:, 0])
#         train_y = list(training[:, 1])

#         model = Sequential()
#         model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
#         model.add(Dropout(0.25))
#         model.add(Dense(128, activation="relu"))
#         model.add(Dropout(0.25))
#         model.add(Dense(64, activation="relu"))
#         model.add(Dropout(0.25))
#         model.add(Dense(len(train_y[0]), activation="softmax"))

#         adam = keras.optimizers.Adam(learning_rate=0.001)
#         model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

#         callback = EarlyStopping(
#             monitor="val_loss",
#             min_delta=0.00001,
#             patience=20,
#             verbose=1,
#             mode="auto",
#             baseline=None,
#             restore_best_weights=False,
#         )

#         model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5   , verbose=1, callbacks=callback)

#         with open("words.json", "w") as f:
#             json.dump(words, f)
#         with open("classes.json", "w") as f:
#             json.dump(classes, f)
#         with open("intents.json", "w") as f:
#             json.dump(data, f)

#         model.save("chatbot_model.h5", overwrite=True)

#     return jsonify({"status": "Model retrained successfully."})


# def parse_txt_data(file_path):
#     intents = []
#     current_tag = ""
#     patterns = []
#     responses = []
#     with open(file_path, "r") as file:
#         for line in file:
#             if line.startswith("Q: "):
#                 current_tag = f"tag_{len(intents)}"
#                 patterns = [line[3:].strip()]
#             elif line.startswith("A: "):
#                 responses = [line[3:].strip()]
#                 intents.append({"tag": current_tag, "patterns": patterns, "responses": responses})
#     return {"intents": intents}

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import os
import json
import keras
import numpy as np
import nltk
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from textblob import TextBlob
import threading
import time
import sys

app = Flask(__name__)

lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = Tokenizer(oov_token="<OOV>")
max_len = 20

words = []
classes = []
documents = []
ignore_letters = ["!", "?", ",", "."]

model = load_model("chatbot_model.h5")
with open("words.json") as f:
    words = json.load(f)
with open("classes.json") as f:
    classes = json.load(f)
with open("intents.json") as f:
    intents_data = json.load(f)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def tokenize_sentence(sentence):
    sentence_words = clean_up_sentence(sentence)
    return " ".join(sentence_words)


def predict_class(sentence, model):
    sentence = tokenize_sentence(sentence)
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    res = model.predict(np.array(padded))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = i["responses"][0]
                break
    else:
        result = "I'm sorry, I don't have this information. You can take the question assistance."
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/manual")
def manual():
    return render_template("user_manual.html")


def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data["msg"]
    feedback = data.get("feedback", "").lower()
    if feedback == "no":
        alternatives = get_alternative_questions(msg, intents_data)
        return jsonify(
            {
                "response": "",
                "alternatives": alternatives,
            }
        )
    elif feedback.isdigit():
        index = int(feedback) - 1
        alternative_questions = get_alternative_questions(msg, intents_data)
        selected_question = alternative_questions[index]
        ints = predict_class(selected_question, model)
        res = get_response(ints, intents_data)
        return jsonify({"response": res})
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents_data)
        return jsonify({"response": res, "alternatives": []})


@app.route("/add", methods=["POST"])
def add_to_dataset():
    data = request.get_json()
    question = data["question"]
    answer = data["answer"]

    with open("dialogs.txt", "a") as f:
        f.write(f"Q: {question}\nA: {answer}\n\n")

    return jsonify({"status": "Question and answer added successfully."})


@app.route("/retrain", methods=["POST"])
def retrain_model():
    data = parse_txt_data("dialogs.txt")
    if "intents" in data and data["intents"]:
        global words, classes, documents, model, tokenizer
        words = []
        classes = []
        documents = []
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                words.extend(word_list)
                documents.append((pattern, intent["tag"]))
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        words = [
            lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters
        ]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        tokenizer.fit_on_texts([doc[0] for doc in documents])
        sequences = tokenizer.texts_to_sequences([doc[0] for doc in documents])
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

        output_empty = [0] * len(classes)
        training = []

        for i, doc in enumerate(documents):
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([padded_sequences[i], output_row])

        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_len))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(classes), activation="softmax"))

        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

        callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00001,
            patience=30,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )

        model.fit(
            np.array(train_x),
            np.array(train_y),
            epochs=500,
            batch_size=8,
            verbose=1,
            callbacks=callback,
            validation_split=0.2,
        )

        with open("words.json", "w") as f:
            json.dump(words, f)
        with open("classes.json", "w") as f:
            json.dump(classes, f)
        with open("intents.json", "w") as f:
            json.dump(data, f)

        model.save("chatbot_model.h5", overwrite=True)

        threading.Thread(
            target=restart_server
        ).start()  # Restart server in a new thread

    return jsonify({"status": "Model retrained successfully."})


def restart_server():
    time.sleep(1)  # Give the server time to send the response to the client
    os.execv(sys.executable, ["python"] + sys.argv)


def parse_txt_data(file_path):
    intents = []
    current_tag = ""
    patterns = []
    responses = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("Q: "):
                current_tag = f"tag_{len(intents)}"
                patterns = [line[3:].strip()]
            elif line.startswith("A: "):
                responses = [line[3:].strip()]
                intents.append(
                    {"tag": current_tag, "patterns": patterns, "responses": responses}
                )
    return {"intents": intents}


if __name__ == "__main__":
    app.run(debug=True)
