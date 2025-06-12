import os
from urllib.parse import urlparse, quote
from dotenv import load_dotenv
load_dotenv()

# Ensure the following variables are set in the environment:
# MLFLOW_DOMAIN, MLFLOW_USERNAME, MLFLOW_PASSWORD, MLFLOW_MODEL_NAME, MLFLOW_MODEL_VERSION

MLFLOW_DOMAIN = os.getenv("MLFLOW_DOMAIN")
MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")

if not all([MLFLOW_DOMAIN, MLFLOW_USERNAME, MLFLOW_PASSWORD, MLFLOW_MODEL_NAME, MLFLOW_MODEL_VERSION]):
    raise EnvironmentError("One or more required MLflow environment variables are missing")

MLFLOW_TRACKING_URI = f"https://{MLFLOW_DOMAIN}"
parsed = urlparse(MLFLOW_TRACKING_URI)
AUTH_URI = parsed._replace(
    netloc=f"{quote(MLFLOW_USERNAME)}:{quote(MLFLOW_PASSWORD)}@{parsed.netloc}"
).geturl()