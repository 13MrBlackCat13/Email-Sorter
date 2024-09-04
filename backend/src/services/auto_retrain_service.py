from src.models.user import UserModel, db
from src.services.classifier import EmailClassifier
from src.services.notification_service import notification_service


class AutoRetrainService:
    def __init__(self, classifier, threshold=100):
        self.classifier = classifier
        self.threshold = threshold

    def check_and_retrain(self, user_id):
        user_model = UserModel.query.filter_by(user_id=user_id).first()
        if user_model and user_model.classification_count >= self.threshold:
            # Получаем новые данные для переобучения
            new_data = self.get_new_training_data(user_id)

            # Загружаем текущую модель пользователя
            self.classifier.load_user_model(user_model.model_path, user_model.vectorizer_path)

            # Переобучаем модель
            accuracies = self.classifier.update_model(new_data['texts'], new_data['labels'])

            # Сохраняем обновленную модель
            self.classifier.save_model(user_model.model_path, user_model.vectorizer_path)

            # Сбрасываем счетчик классификаций
            user_model.classification_count = 0
            db.session.commit()

            # Отправляем уведомление пользователю
            notification_service.notify_model_update(user_id, accuracies)

    def get_new_training_data(self, user_id):
        # Здесь должна быть логика получения новых данных для обучения
        # Например, из базы данных или внешнего источника
        pass


auto_retrain_service = AutoRetrainService(EmailClassifier())