import os
import sys
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir):
    categories = os.listdir(data_dir)
    messages = []
    categories_list = []
    for category in categories:
        for message in os.listdir(os.path.join(data_dir, category)):
            try:
                # Try reading in utf-8
                with open(os.path.join(data_dir, category, message), 'r', encoding='utf-8') as f:
                    messages.append(f.read())
                    categories_list.append(category)
            except UnicodeDecodeError:
                try:
                    # Try reading in utf-16 without BOM
                    with open(os.path.join(data_dir, category, message), 'r', encoding='utf-16le', errors='ignore') as f:
                        messages.append(f.read())
                        categories_list.append(category)
                except UnicodeError:
                    logging.error(f"Error decoding file: {os.path.join(data_dir, category, message)}")

    data = pd.DataFrame({'text': messages, 'category': categories_list})
    return data

def train_model(model_name, X_train_vectorized, y_train):
    if model_name == 'lr':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'svm':
        model = SVC()
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    elif model_name == 'gb':
        model = GradientBoostingClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid model_name")

    model.fit(X_train_vectorized, y_train)
    return model

def test_model(model, X_test_vectorized, y_test):
    y_pred = model.predict(X_test_vectorized)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train and save text classification models')
    parser.add_argument('-train', choices=['all', 'lr', 'svm', 'rf', 'gb'], default='all', help='Specify which model to train')
    args = parser.parse_args()

    data_dir_text = os.path.join(os.getcwd(), 'text')
    data_dir_emails = os.path.join(os.getcwd(), 'emails')

    # Load and preprocess data for text and emails
    data_text = load_data(data_dir_text)
    data_emails = load_data(data_dir_emails)

    # Train/test split for text and emails datasets
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(data_text['text'], data_text['category'], test_size=0.3, random_state=42)
    X_train_emails, X_test_emails, y_train_emails, y_test_emails = train_test_split(data_emails['text'], data_emails['category'], test_size=0.3, random_state=42)

    # Vectorization
    vectorizer_text = TfidfVectorizer()
    vectorizer_emails = TfidfVectorizer()

    X_train_vectorized_text = vectorizer_text.fit_transform(X_train_text)
    X_test_vectorized_text = vectorizer_text.transform(X_test_text)
    X_train_vectorized_emails = vectorizer_emails.fit_transform(X_train_emails)
    X_test_vectorized_emails = vectorizer_emails.transform(X_test_emails)

    models_to_train = ['lr', 'svm', 'rf', 'gb'] if args.train == 'all' else [args.train]

    if not os.path.exists('models'):
        os.makedirs('models')

    def train_and_save(model_name):
        model_text = train_model(model_name, X_train_vectorized_text, y_train_text)
        model_emails = train_model(model_name, X_train_vectorized_emails, y_train_emails)

        accuracy_text = test_model(model_text, X_test_vectorized_text, y_test_text)
        accuracy_emails = test_model(model_emails, X_test_vectorized_emails, y_test_emails)

        logging.info(f'{model_name.upper()} Accuracy for Text: {accuracy_text:.2f}')
        logging.info(f'{model_name.upper()} Accuracy for Emails: {accuracy_emails:.2f}')

        joblib.dump(model_text, f'models/text-model-{model_name}.pkl')
        joblib.dump(model_emails, f'models/emails-model-{model_name}.pkl')
        joblib.dump(vectorizer_text, f'models/text-vectorizer-{model_name}.pkl')
        joblib.dump(vectorizer_emails, f'models/emails-vectorizer-{model_name}.pkl')

    Parallel(n_jobs=-1)(delayed(train_and_save)(model_name) for model_name in models_to_train)
    logging.info("Models trained and saved successfully")

if __name__ == '__main__':
    main()
