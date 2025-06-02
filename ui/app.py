import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageSequence
import threading
import pygame
from ultralytics import YOLO
import time
import mediapipe as mp
from loguru import logger
from shared.utils import extend_bounding_box_area
from shared.driver_behavior_model import DriverBehaviorClassifier


pygame.mixer.init()


def play_audio(file):
    def _play():
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

    threading.Thread(target=_play, daemon=True).start()


class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Alert System")
        self.root.geometry("920x700")
        self.root.configure(bg="#f4f4f4")

        self.init_variables()
        self.init_ui()
        self.load_gif()
        self.load_model()

    def load_model(self):
        self.yolo_model = YOLO("yolo11s.pt")
        # self.classifier = DriverBehaviorClassifier()
        self.classifier = DriverBehaviorClassifier("./models/cnn_based/mobilenet-cpu.h5")

    def init_variables(self):
        # Khởi tạo Mediapipe Face
        self.mp_face = mp.solutions.face_detection

        # model_selection = 0: 0: short range, 1: long range
        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.45)

        self.cap = cv2.VideoCapture(0)
        self.monitoring = False

        self.gif_frames = []
        self.gif_frame_index = 0
        self.is_gif_playing = True

        self.is_playing_warning = False
        self.last_warning_time = 0
        self.warning_interval = 3

        self.last_boxes = []
        self.frame_count = 0
        self.frame_skip = 3
        self.start_time = time.time()

        self.predict_thread = None
        self.frame_for_predict = None
        self.predict_lock = threading.Lock()

        self.sleepy_eye_count = 0
        self.sleepy_yawn_count = 0
        self.lookaway_count = 0
        self.phone_count = 0

        self.using_phone = False

        # Khởi tạo về tình trạng buồn ngủ
        self.start_sleepy_eye_time = None
        self.maximum_sleepy_eye_duration = 10  # giây
        self.warned_3s = False
        self.warned_5s = False
        self.warned_10s = False
        self.is_playing_stop_warning = False # Thông báo dừng xe có đang phát hay không


    def init_ui(self):
        self.init_video_frame()
        self.init_status_panel()
        self.init_stat_panel()
        self.init_buttons()

    def init_video_frame(self):
        self.video_frame = tk.Label(self.root, bg="#ddd")
        self.video_frame.place(x=20, y=20, width=540, height=400)

    def init_status_panel(self):
        status_frame = tk.LabelFrame(self.root, text="Trạng thái tài xế", font=("Helvetica", 14, "bold"), fg="#333")
        status_frame.place(x=580, y=20, width=320, height=100)

        self.status_label = tk.Label(status_frame, text="🟢 Bình thường", font=("Helvetica", 16), fg="green")
        self.status_label.pack(pady=20)

    def init_stat_panel(self):
        stat_frame = tk.LabelFrame(self.root, text="📊 Thống kê cảnh báo", font=("Helvetica", 14))
        stat_frame.place(x=580, y=140, width=320, height=400)

        # Buồn ngủ
        tk.Label(stat_frame, text="🔴 Buồn ngủ", font=("Helvetica", 13, "bold"), fg="red").pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        self.eye_label = tk.Label(
            stat_frame, text="👁️ Mắt buồn ngủ: 0", font=("Segoe UI Emoji", 12), fg="red", anchor="w"
        )
        self.eye_label.pack(fill="x", padx=20, pady=2)

        self.yawn_label = tk.Label(stat_frame, text="😪 Ngáp: 0", font=("Segoe UI Emoji", 12), fg="red", anchor="w")
        self.yawn_label.pack(fill="x", padx=20, pady=2)

        # Mất tập trung
        tk.Label(stat_frame, text="🟠 Mất tập trung", font=("Helvetica", 13, "bold"), fg="orange").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.look_label = tk.Label(
            stat_frame, text="👀 Nhìn hướng khác: 0", font=("Helvetica", 12), fg="orange", anchor="w"
        )
        self.look_label.pack(fill="x", padx=20, pady=2)

        self.phone_label = tk.Label(
            stat_frame, text="📱 Dùng điện thoại: 0", font=("Helvetica", 12), fg="orange", anchor="w"
        )
        self.phone_label.pack(fill="x", padx=20, pady=2)

    def init_buttons(self):
        self.start_icon = ctk.CTkImage(Image.open("./assets/start.png").resize((24, 24)))
        self.stop_icon = ctk.CTkImage(Image.open("./assets/stop.png").resize((24, 24)))

        self.start_button = ctk.CTkButton(
            master=self.root,
            text="Start",
            image=self.start_icon,
            compound="left",
            command=self.start_monitoring,
            fg_color="#4CAF50",
            hover_color="#45A049",
            text_color="white",
            corner_radius=10,
            font=("Arial", 12),
            width=100,
            height=36,
        )
        self.start_button.place(x=600, y=580)

        self.stop_button = ctk.CTkButton(
            master=self.root,
            text="Stop",
            image=self.stop_icon,
            compound="left",
            command=self.stop_monitoring,
            fg_color="#f44336",
            hover_color="#d32f2f",
            text_color="white",
            corner_radius=10,
            font=("Arial", 12),
            width=100,
            height=36,
        )
        self.stop_button.place(x=720, y=580)

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def load_gif(self):
        gif = Image.open("./assets/placeholder.gif")
        self.gif_frames = [ImageTk.PhotoImage(frame.copy().convert("RGBA")) for frame in ImageSequence.Iterator(gif)]
        self.play_gif()

    def play_gif(self):
        if self.is_gif_playing:
            frame = self.gif_frames[self.gif_frame_index]
            self.video_frame.config(image=frame)
            self.gif_frame_index = (self.gif_frame_index + 1) % len(self.gif_frames)
            self.root.after(100, self.play_gif)

    def start_monitoring(self):
        self.monitoring = True
        self.is_gif_playing = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.update_frame()

    def stop_monitoring(self):
        self.monitoring = False
        self.is_gif_playing = True
        self.play_gif()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def set_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def detect_and_crop_face(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                ih, iw, _ = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                x, y, w, h = extend_bounding_box_area(iw, ih, x, y, w, h, extension_ratio=1.2)

                face_crop = frame[y : y + h, x : x + w]
                return face_crop
        return None

    def yolo_detect_phone(self, frame):
        with self.predict_lock:
            results = self.yolo_model.predict(frame, conf=0.6, classes=[67], verbose=False)
            self.last_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                self.last_boxes.append((x1, y1, x2, y2, conf))

            if self.last_boxes:
                self.using_phone = True
                self.set_status("📱 Dùng điện thoại", "orange")
                self.phone_label.config(text=f"📱 Dùng điện thoại: {self.phone_count}")
                now = time.time()
                if now - self.last_warning_time > self.warning_interval:
                    self.phone_count += 1
                    play_audio("./assets/audios/not_use_phone.wav")
                    self.last_warning_time = now
            else:
                self.using_phone = False
                self.last_warning_time = 0

    def run_predict_thread(self):
        self.yolo_detect_phone(self.frame_for_predict)
        self.predict_thread = None

    def reset_warnings(self):
        self.start_sleepy_eye_time = None
        self.warned_3s = False
        self.warned_5s = False
        self.warned_10s = False
    
    def check_eye_closed(self):
        if self.start_sleepy_eye_time is None:
            self.start_sleepy_eye_time = time.time()

        duration = time.time() - self.start_sleepy_eye_time
        audio_path = "./assets/audios/warn_level2.wav" if not self.warned_3s else "./assets/audios/warn_level3.wav"
        is_stop_warning = False  # flag xem có đang cảnh báo stop_car_warning

        if duration >= 6 and not self.warned_5s:
            audio_path = "./assets/audios/stop_car_warning.wav"
            self.warned_5s = True
            is_stop_warning = True
        elif (duration >= 3 and not self.warned_3s) or (self.warned_3s and duration >= 7):
            audio_path = "./assets/audios/warn_level3.wav"
            self.warned_3s = True

        return audio_path, is_stop_warning


    def update_error_counts(self, labels):
        if "sleepy_eye" in labels:
            self.sleepy_eye_count += 1
            self.eye_label.config(text=f"👁️ Mắt buồn ngủ: {self.sleepy_eye_count}")
        if "yawn" in labels:
            self.sleepy_yawn_count += 1
            self.yawn_label.config(text=f"😪 Ngáp: {self.sleepy_yawn_count}")
        if "look_away" in labels:
            self.lookaway_count += 1
            self.look_label.config(text=f"👀 Nhìn hướng khác: {self.lookaway_count}")

    def post_process_predictions(self, labels):
        # Loại bỏ nhãn 'natural'
        labels = [label for label in labels if label != "natural"]
        num_errors = len(labels)

        if num_errors == 0:
            self.reset_warnings()
            self.set_status("🟢 Tình trạng bình thường", "green")
            return

        # Mức cảnh báo mặc định
        status_text = "🟠 Cảnh báo nhẹ"
        status_color = "orange"
        audio_path = "./assets/audios/warn_level1.wav"

        # Phân tích theo nhãn
        if num_errors == 1:
            if "look_away" in labels:
                status_text = "🟠 Mất tập trung"
                status_color = "orange"
                audio_path = "./assets/audios/look_straight.wav"
                play_audio(audio_path)
            elif "rub_eye" in labels:
                status_text = "🟠 Dụi mắt - dấu hiệu mệt mỏi"
                status_color = "orange"
                audio_path = "./assets/audios/warn_level1.wav"
                play_audio(audio_path)
            elif "yawn" in labels:
                status_text = "🟠 Dấu hiệu buồn ngủ nhẹ"
                status_color = "red"
                audio_path = "./assets/audios/warn_level2.wav"
                play_audio(audio_path)

            if "sleepy_eye" in labels:
                if self.start_sleepy_eye_time is None:
                    self.start_sleepy_eye_time = time.time()
                
                # Kiểm tra thời gian buồn ngủ
                logger.debug(f"Time since sleepy eye started: {time.time() - self.start_sleepy_eye_time}")
                audio_path, is_stop_warning = self.check_eye_closed()

                if is_stop_warning:
                    # Nếu đang phát cảnh báo dừng xe rồi
                    if not self.is_playing_stop_warning:
                        # Nếu chưa phát thì phát và set trạng thái đang phát
                        self.is_playing_stop_warning = True

                        def play_and_reset():
                            pygame.mixer.music.load(audio_path)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            self.is_playing_stop_warning = False

                        threading.Thread(target=play_and_reset, daemon=True).start()
                else:
                    # Nếu không phải cảnh báo dừng xe, mà cảnh báo stop_warning đang phát thì đợi nó xong
                    def play_warn3_when_ready():
                        while self.is_playing_stop_warning:
                            time.sleep(0.5)
                        play_audio(audio_path)

                    threading.Thread(target=play_warn3_when_ready, daemon=True).start()
        else:
            if "look_away" in labels:
                status_text = "🟣 Mất tập trung kèm dấu hiệu buồn ngủ"
                status_color = "purple"
            else:
                status_text = "🔴 Buồn ngủ nghiêm trọng"
                status_color = "red"
            audio_path = "./assets/audios/warn_level3.wav"
            play_audio(audio_path)

        # Cập nhật giao diện và âm thanh cảnh báo
        self.set_status(status_text, status_color)
        self.update_error_counts(labels)


    def update_frame(self):
        if self.monitoring:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 400))
                current_time = time.time()

                if current_time - self.start_time >= 3:
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip == 0 and self.predict_thread is None:
                        self.frame_for_predict = frame.copy()
                        self.predict_thread = threading.Thread(target=self.run_predict_thread)
                        self.predict_thread.start()

                for x1, y1, x2, y2, conf in self.last_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Cell phone {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                detect_interval = 1.0  # giây

                if not self.using_phone and (
                    not hasattr(self, "last_detect_time") or (current_time - self.last_detect_time) > detect_interval
                ):
                    face_img = self.detect_and_crop_face(frame)
                    if face_img is not None:
                        labels, confidences = self.classifier.predict_from_image(face_img)
                        self.post_process_predictions(labels)
                        logger.info(f"Detected labels: {labels}, {confidences}")
                    else:
                        logger.info("No face detected")
                        play_audio("./assets/audios/look_straight.wav")

                    self.last_detect_time = current_time  # cập nhật lại thời điểm detect cuối cùng

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_frame.configure(image=img)
                self.video_frame.image = img

            self.root.after(30, self.update_frame)

    def __del__(self):
        self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.mainloop()
