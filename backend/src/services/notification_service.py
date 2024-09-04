from flask_mail import Mail, Message
from src.models.user import User, db

mail = Mail()

class NotificationService:
    @staticmethod
    def send_email(user_id, subject, body):
        user = User.query.get(user_id)
        if user and user.email:
            msg = Message(subject, recipients=[user.email])
            msg.body = body
            mail.send(msg)

    @staticmethod
    def notify_training_complete(user_id, accuracies):
        subject = "Model Training Complete"
        body = f"Your model has finished training. Accuracies: {accuracies}"
        NotificationService.send_email(user_id, subject, body)

    @staticmethod
    def notify_model_update(user_id, accuracies):
        subject = "Model Update Complete"
        body = f"Your model has been updated. New accuracies: {accuracies}"
        NotificationService.send_email(user_id, subject, body)

notification_service = NotificationService()