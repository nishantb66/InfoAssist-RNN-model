def add_to_dataset():
    # Open the file in append mode
    with open("dialogs.txt", "a") as f:
        while True:
            # Ask the user for a question
            question = input("Enter your question (or 'quit' to stop): ")
            if question.lower() == "quit":
                break

            # Ask the user for an answer
            answer = input("Enter the answer: ")

            # Write the question and answer to the file
            f.write(f"Q: {question}\nA: {answer}\n\n")


# Call the function
add_to_dataset()
