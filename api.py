import json
import os
import subprocess
import tempfile
from flask import Flask, request, jsonify, abort
from sklearn.svm import SVC
import joblib
import numpy as np

app = Flask(__name__)

MODELS_CACHE = {}
VECTORIZERS_CACHE = {}
TEXT_VECTORIZATION_CACHE = {}

def load_model_and_vectorizer(model_name):
    # Кеширование моделей и векторизаторов для повышения эффективности
    if model_name not in MODELS_CACHE:
        MODELS_CACHE[model_name] = joblib.load(f'models/{model_name}-model.pkl')
        VECTORIZERS_CACHE[model_name] = joblib.load(f'models/{model_name}-vectorizer.pkl')
    return MODELS_CACHE[model_name], VECTORIZERS_CACHE[model_name]

# Предварительная загрузка моделей и векторизаторов
model_names = ['text-rf', 'emails-rf', 'text-lr', 'emails-lr', 'text-svm', 'emails-svm', 'text-gb', 'emails-gb']
for name in model_names:
    load_model_and_vectorizer(name)

# Define the super token
SUPER_TOKEN = 'devtest'

# Define rate limits for token groups (requests per minute)
RATE_LIMITS = {
    'group1': 30,
    'group2': 60,
    'group3': 180
}

# Load existing tokens from json file (if available)
existing_tokens = {}
if os.path.exists('tokens.json'):
    with open('tokens.json', 'r') as f:
        existing_tokens = json.load(f)


# Helper function to extract text from .eml file
def extract_text_from_eml(eml_file):
    with open(eml_file, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f)
        text = ''
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                text += part.get_payload()
        return text


# Function to validate the token and check rate limits
def validate_token(token):
    if '-dev' in sys.argv:  # Проверяем наличие флага -dev
        return 'dev', None  # Возвращаем 'dev' в качестве группы и None для лимита
    if token == SUPER_TOKEN:
        return 'super', None
    elif token in existing_tokens:
        group = existing_tokens[token]['group']
        limit = RATE_LIMITS.get(group, 0)
        return group, limit
    else:
        return None, None


# Function to get the specified model for text analysis
def get_model(model_name):
    models = {
        'rf': rf_model_text,
        'lr': lr_model_text,
        'svm': svm_model_text,
        'gb': gb_model_text
    }
    return models.get(model_name)


def get_all_models():
    models = OrderedDict()
    models['rf'] = rf_model_text
    models['lr'] = lr_model_text
    models['svm'] = svm_model_text
    models['gb'] = gb_model_text
    return models


@app.route('/api/emailcategory/text', methods=['POST'])
def predict_from_text():
    data = request.get_json()
    token = data.get('token')
    model_name = data.get('model_name', 'rf')  # Default to RandomForest if model_name is not specified
    text = data.get('text')

    # Validate the token and check rate limits
    group, limit = validate_token(token)
    if group is None:
        abort(401, 'Invalid token')

    # Check rate limits
    if limit is not None:
        # Count the number of requests for the token within the last minute
        # (You can use a more sophisticated rate limiting mechanism if needed)
        if group in RATE_LIMITS:
            if RATE_LIMITS[group]['count'] >= limit:
                abort(429, 'Rate limit exceeded')
            RATE_LIMITS[group]['count'] += 1
        else:
            RATE_LIMITS[group] = {'count': 1}

    if model_name == 'all':
        all_models = get_all_models()
        results = []
        for model_name, model in all_models.items():
            X_new = vectorizer_text.transform([text])
            if isinstance(model, SVC):
                distances = model.decision_function(X_new)[0]
                probabilities = 1 / (1 + np.exp(-distances))
            else:
                probabilities = model.predict_proba(X_new)[0]
            categories = model.classes_
            result = {
                'Model': model_name
            }
            for category, probability in zip(categories, probabilities):
                result[category] = round(probability, 2) if probability > 0.01 else 0
            results.append(result)

        # Calculate the average result
        avg_result = {}
        for category in results[0].keys():
            if category != 'Model':  # Skip the 'Model' key
                probabilities = [float(result[category]) for result in results]  # Convert probabilities to float
                avg_result[category] = round(np.mean(probabilities), 2)

        results.append({
            'Model': 'Average',
            **avg_result
        })

        return jsonify(results)

    model = get_model(model_name)
    if model is None:
        abort(400, 'Invalid model_name')

    X_new = vectorizer_text.transform([text])
    if isinstance(model, SVC):
        probabilities = model.decision_function(X_new)[0]
        probabilities = 1 / (1 + np.exp(-probabilities))
    else:
        probabilities = model.predict_proba(X_new)[0]
    categories = model.classes_

    result = {
        'Model': model_name
    }
    for category, probability in zip(categories, probabilities):
        result[category] = round(probability, 2) if probability > 0.01 else 0

    return jsonify([result])


# Route to handle model training
@app.route('/api/emailcategory/train', methods=['POST'])
def train_models():
    data = request.get_json()
    token = data.get('token')

    # Validate the super token
    if token != SUPER_TOKEN:
        abort(401, 'Invalid token')

    try:
        # Run the train.py script to train models
        subprocess.run(['python', 'train.py'], check=True)
        return jsonify({'message': 'Model training started successfully'})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)})


# Route to handle .eml file input
@app.route('/api/emailcategory/eml', methods=['POST'])
def predict_from_eml():
    data = request.form
    token = data.get('token')
    eml_file = request.files.get('file')
    model_name = data.get('model_name', 'rf')  # Default to RandomForest if model_name is not specified
    text_only = data.get('text_only', 'false').lower() == 'true'  # Check if text_only is set to 'true'

    # Validate the token and check rate limits
    group, limit = validate_token(token)
    if group is None:
        abort(401, 'Invalid token')

    # Check rate limits
    if limit is not None:
        # Count the number of requests for the token within the last minute
        # (You can use a more sophisticated rate limiting mechanism if needed)
        if group in RATE_LIMITS:
            if RATE_LIMITS[group]['count'] >= limit:
                abort(429, 'Rate limit exceeded')
            RATE_LIMITS[group]['count'] += 1
        else:
            RATE_LIMITS[group] = {'count': 1}

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, 'temp.eml')
    eml_file.save(temp_file_path)

    # Extract email information from the .eml file
    with open(temp_file_path, 'rb') as f:
        email_content = f.read()
    body = extract_text_from_eml(temp_file_path)
    subject = eml_file.filename  # Use the filename as the subject

    # Extract 'From' information
    msg = email.message_from_file(open(temp_file_path, 'r', encoding='utf-8', errors='ignore'))
    sender = msg["From"] if "From" in msg else ""

    # Remove timestamp from subject
    subject = subject.split(" - ")[0]

    if model_name == 'all':
        all_models = get_all_models()
        results = []
        for model_name, model in all_models.items():
            X_new = vectorizer_text.transform([body])
            if isinstance(model, SVC):
                probabilities = model.decision_function(X_new)[0]
                probabilities = 1 / (1 + np.exp(-probabilities))
            else:
                probabilities = model.predict_proba(X_new)[0]
            categories = model.classes_
            result = {
                'Model': model_name
            }
            for category, probability in zip(categories, probabilities):
                result[category] = round(probability, 2) if probability > 0.01 else 0
            results.append(result)

        # Calculate the average result
        avg_result = {}
        for category in results[0].keys():
            if category != 'Model':  # Skip the 'Model' key
                probabilities = [float(result[category]) for result in results]  # Convert probabilities to float
                avg_result[category] = round(np.mean(probabilities), 2)

        results.append({
            'Model': 'Average',
            **avg_result
        })

        if not text_only:
            # Include the 'Body', 'From', and 'Subject' information in the response if text_only is not set to 'true'
            results[-1].update({
                "Body": body,
                "From": sender,
                "Subject": subject
            })

        return jsonify(results)  # Return the response here

    model = get_model(model_name)
    if model is None:
        abort(400, 'Invalid model_name')

    X_new = vectorizer_text.transform([body])
    if isinstance(model, SVC):
        probabilities = model.decision_function(X_new)[0]
        probabilities = 1 / (1 + np.exp(-probabilities))
    else:
        probabilities = model.predict_proba(X_new)[0]
    categories = model.classes_

    result = {
        'Model': model_name
    }
    for category, probability in zip(categories, probabilities):
        result[category] = round(probability, 2) if probability > 0.01 else 0

    return jsonify([result])


# Helper function to save existing tokens to json file
def save_tokens():
    with open('tokens.json', 'w') as f:
        json.dump(existing_tokens, f)


# Route to handle token addition, editing, and deletion
@app.route('/api/emailcategory/token', methods=['POST', 'PUT', 'DELETE'])
def manage_tokens():
    data = request.get_json()
    super_token = data.get('super_token')
    action = data.get('action')  # 'add', 'edit', or 'delete'
    token = data.get('token')  # Token to add, edit, or delete
    group = data.get('group')  # Group for the token

    # Validate the super token
    if super_token != SUPER_TOKEN:
        abort(401, 'Invalid super_token')

    if action == 'add':
        if not token or not group:
            abort(400, 'Token and group must be provided for addition')

        # Add the token to the existing_tokens dictionary
        existing_tokens[token] = {'group': group}
        save_tokens()
        return jsonify({'message': 'Token added successfully'})

    elif action == 'edit':
        if not token or not group:
            abort(400, 'Token and group must be provided for editing')

        # Check if the token exists in the existing_tokens dictionary
        if token not in existing_tokens:
            abort(404, 'Token not found')

        # Update the group for the token
        existing_tokens[token]['group'] = group
        save_tokens()
        return jsonify({'message': 'Token edited successfully'})

    elif action == 'delete':
        if not token:
            abort(400, 'Token must be provided for deletion')

        # Check if the token exists in the existing_tokens dictionary
        if token not in existing_tokens:
            abort(404, 'Token not found')

        # Delete the token from the existing_tokens dictionary
        del existing_tokens[token]
        save_tokens()
        return jsonify({'message': 'Token deleted successfully'})

    else:
        abort(400, 'Invalid action. Use "add", "edit", or "delete"')


if __name__ == '__main__':
    if '-dev' in sys.argv:
        app.run(debug=True, port=5000)
    else:
        app.run(debug=False, port=5000)  
