
import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import logging
from joblib import Parallel, delayed

from model_utils import train_model, test_model
from data_utils import load_data
from config import Config

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Train and save text classification models')
    parser.add_argument('-data_dir', type=str, default=Config.DATA_DIR, help='Directory containing training data')
    parser.add_argument('-train', choices=['all', 'lr', 'svm', 'rf', 'gb'], default='all', help='Specify which model to train')
    args = parser.parse_args()

    data = load_data(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category'], test_size=0.3, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    models_to_train = ['lr', 'svm', 'rf', 'gb'] if args.train == 'all' else [args.train]

    if not os.path.exists('models'):
        os.makedirs('models')

    def train_and_save(model_name):
        model = train_model(model_name, X_train_vectorized, y_train)
        accuracy = test_model(model, X_test_vectorized, y_test)
        logging.info(f'{model_name.upper()} Accuracy: {accuracy:.2f}')

        joblib.dump(model, f'models/model-{model_name}.pkl')
        joblib.dump(vectorizer, f'models/vectorizer-{model_name}.pkl')

    Parallel(n_jobs=-1)(delayed(train_and_save)(model_name) for model_name in models_to_train)
    logging.info("Models trained and saved successfully")

if __name__ == '__main__':
    main()
