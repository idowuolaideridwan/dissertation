import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from scipy.stats import uniform
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump  # Import joblib for saving and loading models

# Ensure NLTK resources are downloaded (uncomment if not already done)
# nltk.download('stopwords')
# nltk.download('wordnet')

# Setup enhanced stopwords list
additional_stopwords = {'some', 'additional', 'words'}
stop_words = set(stopwords.words('english')).union(additional_stopwords)

# Setup the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Load and preprocess the dataset
df = pd.read_csv('../data/dataset_v2.csv')
df['comment'] = df['comment'].apply(preprocess_text)

# Handle class imbalance
df_majority = df[df['comment_type'] == 'answer']
df_minority = df[df['comment_type'] == 'question']
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df_balanced['comment'], df_balanced['comment_type'], test_size=0.2, random_state=42)

# Set up the pipeline with the vectorizer and logistic regression model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('scaler', StandardScaler(with_mean=False)),  # Use with_mean=False to support sparse matrices
    ('clf', LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
])

# Define the parameter grid as a dictionary
param_distributions = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [1, 5, 10],
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__use_idf': [True, False],
    'clf__C': uniform(loc=0.01, scale=10),  # uniform continuous distribution
    'clf__solver': ['liblinear', 'lbfgs']  # 'saga' if the dataset is large
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50, cv=5, scoring='accuracy', random_state=42)

# Fit RandomizedSearchCV on the training data
random_search.fit(X_train, y_train)

# Print the best parameters and the best score obtained
print("Best parameters:", random_search.best_params_)
print("Best score: {:.2f}".format(random_search.best_score_))

# Save the best model from RandomizedSearchCV
dump(random_search.best_estimator_, 'best_model_random.pkl')
print("Model saved as 'best_model_random.pkl'")

# Evaluate the model on the test set
y_pred = random_search.predict(X_test)
print("Test Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
