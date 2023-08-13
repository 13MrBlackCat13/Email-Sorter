import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

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
                    print(f"Error decoding file: {os.path.join(data_dir, category, message)}")

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

    # Train and test logistic regression model for text
    model_lr_text = train_model('lr', X_train_vectorized_text, y_train_text)
    accuracy_lr_text = test_model(model_lr_text, X_test_vectorized_text, y_test_text)
    print(f'Logistic Regression Accuracy for Text: {accuracy_lr_text:.2f}')
    joblib.dump(model_lr_text, 'models/text-model-lr.pkl')

    # Train and test logistic regression model for emails
    model_lr_emails = train_model('lr', X_train_vectorized_emails, y_train_emails)
    accuracy_lr_emails = test_model(model_lr_emails, X_test_vectorized_emails, y_test_emails)
    print(f'Logistic Regression Accuracy for Emails: {accuracy_lr_emails:.2f}')
    joblib.dump(model_lr_emails, 'models/emails-model-lr.pkl')

    # Train and test support vector machine model for text
    model_svm_text = train_model('svm', X_train_vectorized_text, y_train_text)
    accuracy_svm_text = test_model(model_svm_text, X_test_vectorized_text, y_test_text)
    print(f'SVM Accuracy for Text: {accuracy_svm_text:.2f}')
    joblib.dump(model_svm_text, 'models/text-model-svm.pkl')

    # Train and test support vector machine model for emails
    model_svm_emails = train_model('svm', X_train_vectorized_emails, y_train_emails)
    accuracy_svm_emails = test_model(model_svm_emails, X_test_vectorized_emails, y_test_emails)
    print(f'SVM Accuracy for Emails: {accuracy_svm_emails:.2f}')
    joblib.dump(model_svm_emails, 'models/emails-model-svm.pkl')

    # Train and test random forest model for text
    model_rf_text = train_model('rf', X_train_vectorized_text, y_train_text)
    accuracy_rf_text = test_model(model_rf_text, X_test_vectorized_text, y_test_text)
    print(f'Random Forest Accuracy for Text: {accuracy_rf_text:.2f}')
    joblib.dump(model_rf_text, 'models/text-model-rf.pkl')

    # Train and test random forest model for emails
    model_rf_emails = train_model('rf', X_train_vectorized_emails, y_train_emails)
    accuracy_rf_emails = test_model(model_rf_emails, X_test_vectorized_emails, y_test_emails)
    print(f'Random Forest Accuracy for Emails: {accuracy_rf_emails:.2f}')
    joblib.dump(model_rf_emails, 'models/emails-model-rf.pkl')

    # Train and test gradient boosting model for text
    model_gb_text = train_model('gb', X_train_vectorized_text, y_train_text)
    accuracy_gb_text = test_model(model_gb_text, X_test_vectorized_text, y_test_text)
    print(f'Gradient Boosting Accuracy for Text: {accuracy_gb_text:.2f}')
    joblib.dump(model_gb_text, 'models/text-model-gb.pkl')

    # Train and test gradient boosting model for emails
    model_gb_emails = train_model('gb', X_train_vectorized_emails, y_train_emails)
    accuracy_gb_emails = test_model(model_gb_emails, X_test_vectorized_emails, y_test_emails)
    print(f'Gradient Boosting Accuracy for Emails: {accuracy_gb_emails:.2f}')
    joblib.dump(model_gb_emails, 'models/emails-model-gb.pkl')

    print("Models trained and saved successfully")

if __name__ == '__main__':
    main()
