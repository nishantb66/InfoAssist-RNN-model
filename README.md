
# InfoAssist-RNN-model

## Project Overview
InfoAssist-RNN-model is a chatbot application designed to assist users by answering frequently asked questions (FAQs) and providing support for various queries related to a software product. This project leverages Natural Language Processing (NLP) and deep learning techniques to deliver accurate and contextually relevant responses.

## Project Structure
The project directory is structured as follows:

```
InfoAssist-RNN-model/
├── __pycache__/
├── myenv/
├── static/
│   ├── admin_logo.jpg
│   └── logo.png
├── templates/
│   ├── admin.html
│   ├── index.html
│   ├── login.html
│   └── user_manual.html
├── APP.py
├── chatbot.py
├── chatbotcopy.py
├── classes.json
├── dialogs.txt
├── dialogs3.txt
├── input_QA.py
├── intents.json
├── requirements.txt
├── Retrain.py
└── words.json
```

### Directory Details

- **static/**: Contains static assets such as images (admin_logo.jpg, logo.png).
- **templates/**: HTML templates for rendering web pages (admin.html, index.html, login.html, user_manual.html).
- **APP.py**: Main application file for running the Flask server and handling web routes.
- **chatbot.py**: Contains the chatbot logic and model definitions.
- **chatbotcopy.py**: Backup or alternative version of the chatbot logic.
- **classes.json**: JSON file containing class labels for intents.
- **dialogs.txt**: Text file containing the training data for the chatbot.
- **dialogs3.txt**: Additional or alternative training data.
- **input_QA.py**: Script for inputting questions and answers.
- **intents.json**: JSON file containing the intents and their associated patterns and responses.
- **requirements.txt**: Lists all the dependencies required to run the project.
- **Retrain.py**: Script for retraining the chatbot model.
- **words.json**: JSON file containing tokenized words used in the training data.

## Model Details

### RNN and LSTM Model
The chatbot uses a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to process and understand the context of the queries. The model architecture includes:

1. **Embedding Layer**: Converts words into dense vectors of fixed size.
2. **Bidirectional LSTM Layers**: Captures information from both past and future states.
3. **Dropout Layers**: Reduces overfitting by randomly setting a fraction of input units to 0 at each update during training.
4. **Dense Layers**: Fully connected layers with ReLU activation.
5. **Softmax Layer**: Outputs the probability distribution over the classes.

### Training and Performance
The model is trained on a dataset of FAQs (`dialogs.txt`) using categorical cross-entropy loss and Adam optimizer. Early stopping is used to prevent overfitting. 

- **Accuracy**: The model achieves an accuracy of 82% on the training data.
- **Loss**: The training loss is 0.83.

### Capabilities
- **Answering FAQs**: Provides accurate responses to frequently asked questions.
- **Alternative Questions**: Suggests alternative questions if the initial response is not satisfactory.
- **Sentiment Analysis**: Analyzes user feedback to improve responses.
- **Retraining**: Supports retraining with new data to improve accuracy and relevance.

### Scope for Improvements
1. **Data Augmentation**: Increase the dataset size by adding more diverse and comprehensive questions and answers.
2. **Hyperparameter Tuning**: Further optimize learning rate, batch size, and number of epochs.
3. **Advanced NLP Techniques**: Implement transformers or attention mechanisms for better context understanding.
4. **User Feedback Loop**: Continuously improve the model based on user interactions and feedback.
5. **Multilingual Support**: Extend the model to support multiple languages.

## Installation and Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/InfoAssist-RNN-model.git
   cd InfoAssist-RNN-model
   ```

2. **Create a virtual environment**:
   ```sh
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scriptsctivate`
   ```

3. **Install the dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```sh
   python APP.py
   ```

5. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. **Ask a Question**:
   - Type your question in the input field on the homepage and submit to get an answer.
2. **Login as Admin**:
   - Access the admin panel to manage questions and answers.
3. **User Manual**:
   - Refer to the user manual for detailed instructions on using the chatbot.

## Contribution
1. **Fork the repository**.
2. **Create a new branch**:
   ```sh
   git checkout -b feature-branch
   ```
3. **Make your changes**.
4. **Commit your changes**:
   ```sh
   git commit -m "Add some feature"
   ```
5. **Push to the branch**:
   ```sh
   git push origin feature-branch
   ```
6. **Create a pull request**.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or need further assistance.

Happy Coding! 🚀
