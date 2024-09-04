from src.api import create_app
from config.config import Config
from src.services.data_processor import create_directory_structure

app = create_app(Config)

if __name__ == '__main__':
    create_directory_structure()
    app.run(debug=True)