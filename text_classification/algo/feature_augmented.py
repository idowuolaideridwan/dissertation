import os
import joblib
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load your own logistic regression model
def load_logistic_regression_model(model_path):
    return joblib.load(model_path)

# Rule-based classifier function
def rule_based_classifier(text):
    question_words = {'what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are', 'can',
                      'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                      "can't", "couldn't", "wouldn't", "won't", "shouldn't", '?'}
    words = set(nltk.word_tokenize(text.lower()))
    return 'question' if question_words.intersection(words) else 'non-question'

def augment_features(dataframe, text_column):
    dataframe['rule_based_output'] = dataframe[text_column].apply(rule_based_classifier)
    dataframe['rule_based_output'] = pd.Categorical(dataframe['rule_based_output']).codes
    return dataframe


from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for rule-based classification

# Custom transformer for rule-based classification
class RuleBasedClassifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply rule-based classifier to each comment in X
        result = []
        for comment in X:
            result.append(rule_based_classifier(comment))
        result = np.array(result).reshape(-1, 1)

        return result


# Train a logistic regression model with augmented features
def train_model_with_augmented_features(data):
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(df[['comment']], df['comment_type'], test_size=0.25,
                                                        random_state=42)

    column_transformer = ColumnTransformer(
        [('tfidf', CountVectorizer(), 'comment')],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('features', column_transformer),
        ('augmentation', RuleBasedClassifier()),  # Rule-based classifier as custom transformer
        ('classifier', load_logistic_regression_model('models\logistic_regression_model.pkl'))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


# Classify new sentences using the trained model with augmented features
def classify_new_sentence(model, text):
    new_data = pd.DataFrame({'text': [text]})
    new_data = augment_features(new_data, 'text')
    prediction = model.predict(new_data)
    return prediction[0]

# Main function for interaction
def main():
    nltk.download('punkt')

    # Example dataset
    data = {
        'comment': ["What time is it?", "This is a book.", "How do you do?", "It is raining."],
        'comment_type': [1, 0, 1, 0]  # 1 for question, 0 for not a question
    }

    # Train the logistic regression model with augmented features
    model = train_model_with_augmented_features(data)

    # Input and classify new sentences
    while True:
        sentence = input("Enter a sentence to classify or type 'exit' to quit: ")
        if sentence.lower() == 'exit':
            break

        classification_result = classify_new_sentence(model, sentence)
        print("Classification Result:", classification_result)

if __name__ == "__main__":
    main()
