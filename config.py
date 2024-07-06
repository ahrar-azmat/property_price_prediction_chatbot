import os
from dotenv import load_dotenv
import logging

load_dotenv('settings.env')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger.debug(f"UPLOAD_FOLDER set to {UPLOAD_FOLDER}")
logger.debug("Configuration loaded successfully")
