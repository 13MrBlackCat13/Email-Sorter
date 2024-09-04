from flask import Blueprint, request, jsonify, send_file, render_template
from src.services.classifier import EmailClassifier
from src.api.auth import token_required
from src.models.user import User, UserModel, db
from src.utils.logger import api_logger
import os
from src.services.notification_service import notification_service
from src.services.auto_retrain_service import auto_retrain_service
from src.services.data_processor import create_directory_structure, process_dataset, process_email

api_bp = Blueprint('api', __name__)
classifier = EmailClassifier()


@api_bp.route('/classify', methods=['POST'])
@token_required
def classify_email(current_user):
    api_logger.info(f"Classification request from user {current_user.username}")
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    user_model = UserModel.query.filter_by(user_id=current_user.id).first()
    if user_model:
        classifier.load_model(user_model.model_path)

    results = classifier.classify(data['text'])

    # Увеличиваем счетчик классификаций
    if user_model:
        user_model.classification_count += 1
        db.session.commit()

        # Проверяем, нужно ли переобучить модель
        auto_retrain_service.check_and_retrain(current_user.id)

    return jsonify(results)


@api_bp.route('/train', methods=['POST'])
@token_required
def train_model(current_user):
    api_logger.info(f"Training request from user {current_user.username}")
    dataset_path = request.json.get('dataset_path')
    if not dataset_path:
        return jsonify({"error": "No dataset path provided"}), 400

    accuracies = classifier.train(dataset_path)

    model_path = f'models/user_{current_user.id}_model.joblib'
    classifier.save_model(model_path)

    user_model = UserModel.query.filter_by(user_id=current_user.id).first()
    if user_model:
        user_model.model_path = model_path
    else:
        new_user_model = UserModel(user_id=current_user.id, model_name='ensemble',
                                   model_path=model_path)
        db.session.add(new_user_model)

    db.session.commit()

    notification_service.notify_training_complete(current_user.id, accuracies)

    return jsonify({"accuracies": accuracies})


@api_bp.route('/update_model', methods=['POST'])
@token_required
def update_model(current_user):
    api_logger.info(f"Model update request from user {current_user.username}")
    data = request.json
    if not data or 'texts' not in data or 'labels' not in data:
        return jsonify({"error": "Invalid training data"}), 400

    user_model = UserModel.query.filter_by(user_id=current_user.id).first()
    if not user_model:
        return jsonify({"error": "No existing model found for this user"}), 404

    classifier.load_model(user_model.model_path)
    accuracies = classifier.update_model(data['texts'], data['labels'])

    classifier.save_model(user_model.model_path)

    return jsonify({"accuracies": accuracies})


@api_bp.route('/export_model', methods=['GET'])
@token_required
def export_model(current_user):
    api_logger.info(f"Model export request from user {current_user.username}")
    user_model = UserModel.query.filter_by(user_id=current_user.id).first()
    if not user_model:
        return jsonify({"error": "No model found for this user"}), 404

    return send_file(user_model.model_path, as_attachment=True)


@api_bp.route('/import_model', methods=['POST'])
@token_required
def import_model(current_user):
    api_logger.info(f"Model import request from user {current_user.username}")
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400

    model_file = request.files['model']
    model_path = f'models/user_{current_user.id}_imported_model.joblib'
    model_file.save(model_path)

    try:
        classifier.load_model(model_path)
    except Exception as e:
        os.remove(model_path)
        return jsonify({"error": f"Invalid model file: {str(e)}"}), 400

    user_model = UserModel.query.filter_by(user_id=current_user.id).first()
    if user_model:
        user_model.model_path = model_path
    else:
        new_user_model = UserModel(user_id=current_user.id, model_name='imported',
                                   model_path=model_path)
        db.session.add(new_user_model)

    db.session.commit()

    return jsonify({"message": "Model imported successfully"})


@api_bp.route('/process_dataset', methods=['POST'])
@token_required
def process_dataset_route(current_user):
    api_logger.info(f"Dataset processing request from user {current_user.username}")
    raw_data_dir = request.json.get('raw_data_dir', 'data/raw')
    processed_data_dir = request.json.get('processed_data_dir', 'data/processed')

    if not os.path.exists(raw_data_dir):
        return jsonify({"error": "Raw data directory does not exist"}), 400

    try:
        process_dataset(raw_data_dir, processed_data_dir)
        return jsonify({"message": "Dataset processed successfully"})
    except Exception as e:
        return jsonify({"error": f"Error processing dataset: {str(e)}"}), 500


@api_bp.route('/create_directory_structure', methods=['POST'])
@token_required
def create_directory_structure_route(current_user):
    api_logger.info(f"Directory structure creation request from user {current_user.username}")
    try:
        create_directory_structure()
        return jsonify({"message": "Directory structure created successfully"})
    except Exception as e:
        return jsonify({"error": f"Error creating directory structure: {str(e)}"}), 500


@api_bp.route('/train_general_model', methods=['POST'])
@token_required
def train_general_model(current_user):
    api_logger.info(f"General model training request from user {current_user.username}")
    dataset_path = request.json.get('dataset_path', 'data/processed')

    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset path does not exist"}), 400

    accuracies = EmailClassifier.train_general_model(dataset_path)

    notification_service.notify_training_complete(current_user.id, accuracies)

    return jsonify({"accuracies": accuracies})

@api_bp.route('/classify_general', methods=['POST'])
@token_required
def classify_general(current_user):
    api_logger.info(f"General classification request from user {current_user.username}")
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    general_classifier = EmailClassifier.load_general_model()
    results = general_classifier.classify(data['text'])

    return jsonify(results)

@api_bp.route('/stats', methods=['GET'])
@token_required
def get_stats(current_user):
    api_logger.info(f"Stats request from user {current_user.username}")
    total_users = User.query.count()
    total_models = UserModel.query.count()
    user_model = UserModel.query.filter_by(user_id=current_user.id).first()

    stats = {
        "total_users": total_users,
        "total_models": total_models,
        "user_has_model": user_model is not None
    }

    if user_model:
        classifier.load_model(user_model.model_path)
        stats["model_performance"] = classifier.get_model_performance()

    return jsonify(stats)

@api_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@api_bp.route('/sort', methods=['GET'])
def sort():
    return render_template('sort.html')

@api_bp.route('/get_emails', methods=['GET'])
def get_emails():
    raw_data_dir = 'data/raw'
    emails = []
    for category in os.listdir(raw_data_dir):
        category_path = os.path.join(raw_data_dir, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                emails.append({
                    'filename': filename,
                    'category': category,
                    'content': content[:200] + '...'  # Preview of content
                })
    return jsonify(emails)

@api_bp.route('/classify_email', methods=['POST'])
def classify_email():
    data = request.json
    if not data or 'content' not in data:
        return jsonify({"error": "No content provided"}), 400

    processed_content = process_email(data['content'])
    classifier = EmailClassifier.load_general_model()
    results = classifier.classify(processed_content)

    return jsonify(results)

@api_bp.route('/move_email', methods=['POST'])
def move_email():
    data = request.json
    if not data or 'filename' not in data or 'from_category' not in data or 'to_category' not in data:
        return jsonify({"error": "Invalid data provided"}), 400

    from_path = os.path.join('data/raw', data['from_category'], data['filename'])
    to_path = os.path.join('data/raw', data['to_category'], data['filename'])

    if not os.path.exists(from_path):
        return jsonify({"error": "Source file does not exist"}), 400

    os.rename(from_path, to_path)
    return jsonify({"message": "Email moved successfully"})