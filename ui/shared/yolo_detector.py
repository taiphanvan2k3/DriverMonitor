from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Model loaded from {self.class_names}")

    def predict_from_path(self, image_path, conf=0.25, show=False):
        results = self.model(image_path, conf=conf)
        return self._process_results(results[0], conf, show)

    def predict_from_image(self, frame, conf=0.25, show=False):
        results = self.model(frame, conf=conf)
        return self._process_results(results[0], conf, show)

    def _process_results(self, result, conf, show):
        if show:
            result.show()

        labels = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = self.class_names[cls_id] if self.class_names else cls_id

            if label == "phone":
                print(f"Detected phone with confidence: {conf_score:.2f}")

            if conf_score >= conf:
                labels.append(label)

        # Lọc labels để loại bỏ các nhãn trùng lặp
        labels = list(set(labels))
        return labels
