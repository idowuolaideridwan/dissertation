from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pymysql
import pymysql.cursors
import random
import string
from functools import wraps
import bcrypt
import logging
import datetime
import torch

import os
import joblib
import nltk

app = Flask(__name__)
app.secret_key = b'\x00\xdc8\xfa\xb1\xd7\x06\x96\x02\xdb<F@7\xf0\xf3\xbf$\x8cb\x94w\xe8\xa3'

# Path to the folder containing saved models
models_folder_path = 'text_classification/algo/'

# Load the rule-based classifier
def rule_based_classifier(text):
    question_words = {'what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are', 'can',
                      'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                      "can't", "couldn't", "wouldn't", "won't", "shouldn't", '?'}
    words = set(nltk.word_tokenize(text.lower()))
    return 'question' if question_words.intersection(words) else 'non-question'

# Load the logistic regression model
def load_model(model_path):
    return joblib.load(model_path)

# Classify a sentence using the logistic regression model
def logistic_regression_classifier(model, text):
    prediction = model.predict([text])
    return prediction[0]

# Determine the classification of text using a hybrid approach
def hybrid_classifier(models, text):
    if rule_based_classifier(text) == 'question':
        return 'question'
    else:
        model_name = 'logistic_regression_model'  # Example, adjust based on your model naming
        if model_name in models:
            return logistic_regression_classifier(models[model_name], text)
        else:
            return 'non-question'

# Load all models from the specified folder
def load_models(models_folder_path):
    models = {}
    for filename in os.listdir(models_folder_path):
        if filename.endswith('.pkl'):
            model_path = os.path.join(models_folder_path, filename)
            model_name = filename.split('.pkl')[0]
            models[model_name] = load_model(model_path)
    return models

# Load all models
models = load_models(models_folder_path)

# Define the API endpoint for classifying sentences
@app.route('/classify', methods=['POST'])
def classify_sentence():
    sentence = request.json.get('sentence', '')
    classification_result = hybrid_classifier(models, sentence)
    return jsonify({'classification_result': classification_result})

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def generate_random_alphanumeric(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=seconds))

def hash_password(password):
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)
    return hashed_password.decode('utf-8')

def check_password(stored_password, provided_password):
    stored_password_bytes = stored_password.encode('utf-8')
    provided_password_bytes = provided_password.encode('utf-8')
    return bcrypt.checkpw(provided_password_bytes, stored_password_bytes)

def get_db_connection():
    try:
        return pymysql.connect(host='testingcicdserverdb.c1uehec5adgq.us-east-1.rds.amazonaws.com', user='admin', password='testing2day', database='vbls', cursorclass=pymysql.cursors.DictCursor)
    except pymysql.MySQLError as e:
        print(f"The error '{e}' occurred")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    connection = get_db_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                if user and check_password(user['password'], password):
                    session['user_id'] = user['id']
                    return redirect(url_for('video_learning'))
            flash('Invalid credentials. Please try again.', 'error')
        finally:
            connection.close()
    else:
        flash('Database connection failed.', 'error')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

def get_comments():
    connection = get_db_connection()
    comments = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT content, start_time, end_time FROM video_comments ORDER BY id DESC LIMIT 10")
                comments = cursor.fetchall()  # Fetches all rows from the last executed statement
        except pymysql.MySQLError as e:
            print(f"The error '{e}' occurred")
        finally:
            connection.close()
    return comments

@app.route('/filter_comments', methods=['GET'])
def filter_comments():
    comments = get_comments()
    comment_type = request.args.get('type')  # Get the type from query parameters
    if comment_type:
        filtered_comments = [comment for comment in comments if comment['type'] == comment_type]
        return jsonify(filtered_comments)
    else:
        return jsonify(comments)

@app.route('/add_comment', methods=['POST'])
def add_comment():
    comment_content = request.form['comment']
    start_time = request.form.get('startTime', type=int)
    end_time = request.form.get('endTime', type=int)
    random_string = generate_random_alphanumeric(10)

    # Perform toxic classification
    toxic_type = 0

    # Perform Question and Answer Classification
    qa_type = 1

    connection = get_db_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                sql = "INSERT INTO video_comments (comment_id, content, qa_type, toxic_type, start_time, end_time) " \
                      "VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (random_string, comment_content, qa_type, toxic_type, start_time, end_time))
                connection.commit()
            return jsonify({'message': 'Comment added successfully'}), 200
        except Exception as e:
            connection.rollback()
            logging.error(f"Failed to add comment: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            connection.close()
    else:
        logging.error("Database connection failed")
        return jsonify({'error': 'Database connection failed'}), 500

@app.route('/video_learning')
@login_required
def video_learning():
    comments = get_comments()
    for comment in comments:
        comment['formatted_start_time'] = seconds_to_hms(comment['start_time'])
        comment['formatted_end_time'] = seconds_to_hms(comment['end_time'])
    return render_template('video_learning.html', comments=comments)

if __name__ == '__main__':
    app.run(debug=True)
