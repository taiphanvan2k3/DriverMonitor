import sys
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def extend_bounding_box_area(iw, ih, x, y, w, h, extension_ratio=1.2):
    cx = x + w / 2
    cy = y + h / 2
    nw = w * extension_ratio
    nh = h * extension_ratio

    nx = int(max(cx - nw / 2, 0))
    ny = int(max(cy - nh / 2, 0))
    nw = int(min(nw, iw - nx))
    nh = int(min(nh, ih - ny))
    return nx, ny, nw, nh

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # Đọc base64 frame từ UI
    import base64
    import numpy as np

    data = base64.b64decode(line)
    np_arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    ih, iw, _ = frame.shape
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        x, y, w, h = extend_bounding_box_area(iw, ih, x, y, w, h)

        # In ra toạ độ dưới dạng: x y w h (cách nhau bằng dấu cách)
        print(f"{x} {y} {w} {h}")
    else:
        # Nếu không detect được mặt, in ra 0 0 0 0 hoặc chuỗi rỗng
        print("0 0 0 0")

    sys.stdout.flush()
