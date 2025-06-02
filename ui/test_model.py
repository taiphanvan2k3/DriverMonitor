import sys
from loguru import logger
from shared.driver_behavior_model import DriverBehaviorClassifier

# Check version python
logger.info("Python version:", sys.version)

classifier = DriverBehaviorClassifier(model_path="./models/cnn_based/mobilenetv2_model_final.h5")

labels, scores = classifier.predict_from_path("image03.png")
logger.info(f"Labels: {labels}")
logger.info(f"Confidence scores: {scores}")
