from flask import Flask
from src.api.routes import api_bp
import os

def create_app(config_object):
    app = Flask(__name__,
                template_folder=os.path.abspath('src/templates'),
                static_folder=os.path.abspath('src/static'))
    app.config.from_object(config_object)

    app.register_blueprint(api_bp)

    return app