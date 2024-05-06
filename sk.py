import pandas as pd

# Load the data
df = pd.read_csv('text_classification/data/discourse/extracted_questions_answers.csv')

# Remove duplicates
df = df.drop_duplicates(subset=['comment'])

# Drop rows with missing values
df = df.dropna()

# Save the cleaned data
df.to_csv('path_to_cleaned_dataset.csv', index=False)

# Optional: Print information about the cleaned data
print("Data cleaned successfully!")
print("Remaining rows after cleaning:", len(df))
