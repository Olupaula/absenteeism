from django.apps import AppConfig
from django.conf import settings
import joblib


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    MODEL_FILE = settings.MODELS/"DecisionTree.joblib"
    model = joblib.load(MODEL_FILE)
