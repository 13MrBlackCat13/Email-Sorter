from flask import Flask
from src.api.routes import api_bp
import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app(config_object):
    app = Flask(__name__,
                template_folder=os.path.abspath('src/templates'),
                static_folder=os.path.abspath('src/static'))
    app.config.from_object(config_object)
    db.init_app(app)
    app.register_blueprint(api_bp)

    return app