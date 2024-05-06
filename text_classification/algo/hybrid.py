import os
import joblib
# import nltk

from rule_based_classifier import rule_based_classifier

def load_model(model_path):
    """ Load a pre-trained model from the specified file. """
    return joblib.load(model_path)

def logistic_regression_classifier(model, text):
    """ Classify a sentence using the loaded logistic regression model. """
    prediction = model.predict([text])
    return prediction[0]

def hybrid_classifier(models, text):
    """ Determine the classification of text using a hybrid approach. """
    # First use the rule-based classifier
    if rule_based_classifier(text) == 'question':
        return 'question'
    else:
        # Use the logistic regression model if available
        model_name = 'logistic_regression'  # Example, adjust based on your model naming
        if model_name in models:
            return logistic_regression_classifier(models[model_name], text)
        else:
            return "Statement"

def main():
    # Path to the folder containing saved models
    models_folder_path = 'models/'

    # Load all .pkl models from the specified folder
    models = {}
    for filename in os.listdir(models_folder_path):
        if filename.endswith('.pkl'):
            model_path = os.path.join(models_folder_path, filename)
            model_name = filename.split('.pkl')[0]
            models[model_name] = load_model(model_path)

    # Input sentence to classify
    sentence = input("Enter a sentence to classify: ")
    # Get the classification result using the hybrid classifier
    classification_result = hybrid_classifier(models, sentence)
    print("Classification Result:", classification_result)

if __name__ == "__main__":
    # nltk.download('punkt')  # Ensure required NLTK resources are downloaded
    main()
