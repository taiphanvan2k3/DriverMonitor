import sys
from loguru import logger
from shared.driver_behavior_model import DriverBehaviorClassifier

# Check version python
logger.info("Python version:", sys.version)

classifier = DriverBehaviorClassifier()

labels, scores = classifier.predict("image06.png")
logger.info(f"Labels: {labels}")
logger.info(f"Confidence scores: {scores}")
