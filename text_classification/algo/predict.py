# classify.py
import joblib


def load_model(model_path):
    """ Load the pre-trained model from a specified path. """
    model = joblib.load(model_path)
    return model


def classify_sentence(model, sentence):
    """ Classify a sentence using the loaded model. """
    prediction = model.predict([sentence])
    return prediction[0]


def main():
    # Path to the saved model
    model_path = 'best_model_random.pkl'

    # Load the model
    model = load_model(model_path)

    # Input sentence to classify
    sentence = input("Enter a sentence to classify: ")

    # Classify the given sentence
    predicted_class = classify_sentence(model, sentence)
    print(f"The predicted class for the input sentence is: {predicted_class}")


if __name__ == "__main__":
    main()
