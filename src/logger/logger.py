import logging
import os
from datetime import datetime

# Create Logs Directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log File Name
LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)

# Configure Logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

# Logger Object
logger = logging.getLogger("CreditCardFraudDetection")

logger.info("Logger initialized successfully.")