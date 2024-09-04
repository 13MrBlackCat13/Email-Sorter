from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np


class EmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            'lr': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
            'svm': OneVsRestClassifier(LinearSVC()),
            'rf': RandomForestClassifier(n_estimators=100),
            'gb': GradientBoostingClassifier(n_estimators=100)
        }
        self.categories = [
            'Входящие', 'Рассылки', 'Социальные сети', 'Чеки_Квитанции',
            'Новости', 'Доставка', 'Госписьма', 'Учёба', 'Игры',
            'Spam_Мошенничество', 'Spam_Обычный'
        ]

    def load_data(self, dataset_path):
        texts = []
        labels = []
        for category in self.categories:
            category_path = os.path.join(dataset_path, category.replace('_', '/'))
            for filename in os.listdir(category_path):
                with open(os.path.join(category_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(category)
        return texts, labels

    def train(self, dataset_path):
        texts, labels = self.load_data(dataset_path)
        X = self.vectorizer.fit_transform(texts)
        y = [self.categories.index(label) for label in labels]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        accuracies = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies[name] = accuracy_score(y_test, y_pred)

        return accuracies

    def classify(self, text):
        X = self.vectorizer.transform([text])
        results = {}
        for name, model in self.models.items():
            probabilities = model.predict_proba(X)[0]
            sorted_indices = np.argsort(probabilities)[::-1]
            model_results = []
            for idx in sorted_indices[:3]:
                category = self.categories[idx]
                probability = probabilities[idx]
                if probability > 0.1:
                    model_results.append((category, probability))
            results[name] = model_results

        # Ensemble prediction
        ensemble_probs = np.mean([model.predict_proba(X)[0] for model in self.models.values()], axis=0)
        sorted_indices = np.argsort(ensemble_probs)[::-1]
        ensemble_results = []
        for idx in sorted_indices[:3]:
            category = self.categories[idx]
            probability = ensemble_probs[idx]
            if probability > 0.1:
                ensemble_results.append((category, probability))
        results['ensemble'] = ensemble_results

        return results

    def save_model(self, model_path):
        joblib.dump((self.vectorizer, self.models), model_path)

    def load_model(self, model_path):
        self.vectorizer, self.models = joblib.load(model_path)

    @classmethod
    def train_general_model(cls, dataset_path):
        classifier = cls()
        accuracies = classifier.train(dataset_path)

        # Сохраняем каждую модель отдельно
        for model_name, model in classifier.models.items():
            model_path = f'models/general_model_{model_name}.joblib'
            joblib.dump((classifier.vectorizer, model), model_path)

        # Сохраняем векторизатор отдельно
        vectorizer_path = 'models/general_vectorizer.joblib'
        joblib.dump(classifier.vectorizer, vectorizer_path)

        return accuracies

    @classmethod
    def load_general_model(cls):
        classifier = cls()
        vectorizer_path = 'models/general_vectorizer.joblib'
        classifier.vectorizer = joblib.load(vectorizer_path)

        for model_name in classifier.models.keys():
            model_path = f'models/general_model_{model_name}.joblib'
            _, classifier.models[model_name] = joblib.load(model_path)

        return classifier

    def update_model(self, texts, labels):
        X = self.vectorizer.transform(texts)
        y = [self.categories.index(label) for label in labels]

        accuracies = {}
        for name, model in self.models.items():
            model.partial_fit(X, y, classes=np.unique(y))
            y_pred = model.predict(X)
            accuracies[name] = accuracy_score(y, y_pred)

        return accuracies

    def get_model_performance(self):
        return {name: model.score for name, model in self.models.items()}