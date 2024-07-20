import os
import json
import numpy as np
import tensorflow as tf
import nltk
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Initialize variables
lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = Tokenizer(oov_token="<OOV>")
max_len = 25

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

nltk.download("punkt")
nltk.download("wordnet")


# Function to parse the .txt file
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


# Load and parse your dataset
data = parse_txt_data("dialogs.txt")

if "intents" in data and data["intents"]:
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
        lemmatizer.lemmatize(w.lower()) for w in words if w not in ["!", "?", ",", "."]
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
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    if os.path.exists("chatbot_model.h5"):
        model = load_model("chatbot_model.h5")
    else:
        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_len))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(classes), activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Adjusted learning rate
        model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

        callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00001, patience=30, verbose=1, mode="auto"
        )

        model.fit(
            train_x,
            train_y,
            epochs=500,  # Increased epochs
            batch_size=8,  # Adjusted batch size
            verbose=1,
            callbacks=[callback],
        )

        with open("words.json", "w") as f:
            json.dump(words, f)
        with open("classes.json", "w") as f:
            json.dump(classes, f)
        with open("intents.json", "w") as f:
            json.dump(data, f)

        model.save("chatbot_model.h5", overwrite=True)


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


print(
    "\nWelcome to the INFOASSIST Bot! Start typing to ask me a question. Type 'quit' to exit.\n"
)

while True:
    print("\nYou: ", end="")
    user_input = input()
    if user_input.lower() == "quit":
        break
    ints = predict_class(user_input, model)
    res = get_response(ints, data)
    print("ChatBot: ", res)

    print("\nWas the answer correct? (yes/no): ", end="")
    feedback = input()

    if feedback.lower() == "no":
        print("\nHere are some other questions that might help:\n")

        all_questions = [
            pattern for intent in data["intents"] for pattern in intent["patterns"]
        ]

        tfidf_matrix = vectorizer.fit_transform([user_input] + all_questions)

        cosine_similarities = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()

        top_question_indices = cosine_similarities.argsort()[:-6:-1]
        for i, index in enumerate(top_question_indices):
            print(f"{i + 1}. {all_questions[index]}")

        print(
            "\nEnter the question number from above for which you want the answer to."
        )

        user_choice = input().strip()
        if user_choice.isdigit():
            selected_index = int(user_choice) - 1
            if 0 <= selected_index < len(top_question_indices):
                selected_question = all_questions[top_question_indices[selected_index]]
                selected_ints = predict_class(selected_question, model)
                selected_res = get_response(selected_ints, data)
                print(f"ChatBot: {selected_res}")
            else:
                print("Invalid number. Please try again.")
        elif user_choice.lower() == "keywords":
            print(
                "\nType the main keywords of your question if your required question is not listed above:"
            )
        else:
            print("Invalid input. Please try again.")
