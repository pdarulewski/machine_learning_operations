import os
from pathlib import Path

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'mlops'))

MODELS_PATH = os.path.join(MODULE_PATH, 'models')
Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
