import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from loguru import logger

class DriverBehaviorClassifier:
    def __init__(self, model_path='./models/cnn_based/mobilenetv2_model_final.h5', threshold=0.5):
        self.threshold = threshold
        self.class_labels = ['natural', 'sleepy_eye', 'yawn', 'rub_eye', 'look_away']

        if os.path.exists(model_path):
            logger.info(f"Loading full model from: {model_path}")
            self.model = load_model(model_path)
        else:
            # Fallback to loading weights
            weights_path = './models/cnn_based/model_finetune_checkpoint.weights.h5'
            logger.info(f"Loading weights from: {weights_path}")
            self.model = self.build_model()
            self.model.load_weights(weights_path)

    def build_model(self, input_shape=(224, 224, 3), num_classes=5):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"[ERROR] Could not load image: {image_path}")

        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)[0]
        predicted_labels = []
        confidences = {}

        for i, label in enumerate(self.class_labels):
            confidence = predictions[i]
            confidences[label] = confidence
            if confidence >= self.threshold:
                predicted_labels.append(label)

        if not predicted_labels:
            max_index = np.argmax(predictions)
            fallback_label = self.class_labels[max_index]
            logger.info(f"No label >= threshold ({self.threshold}). Fallback: {fallback_label}")
            predicted_labels.append(fallback_label)

        return predicted_labels, confidences