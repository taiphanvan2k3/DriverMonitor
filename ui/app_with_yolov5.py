from pathlib import Path
import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageSequence
import threading
import pygame
import time
import mediapipe as mp
from telegram.telegram_helper import send_driver_drowsiness_alert_from_frame
import torch
import os, pathlib

pygame.mixer.init()

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
)

# Check xem đã có folder yolov5 chưa
repo_dir = Path(__file__).parent / "yolov5"
if not repo_dir.exists():
    # Chạy lệnh git clone
    import subprocess

    print("========= Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(repo_dir)], check=True)
    print("========= YOLOv5 repository cloned. ==========")

    # Cài đặt yêu cầu
    print("========= Installing YOLOv5 dependencies...")
    subprocess.run(
        ["pip", "install", "-r", str(repo_dir / "requirements.txt")],
        check=True,
    )
    print("========= YOLOv5 dependencies installed. ==========")


def play_audio(file):
    def _play():
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

    threading.Thread(target=_play, daemon=True).start()


class _PosixPathPatcher:
    def __enter__(self):
        self._orig = getattr(pathlib, "PosixPath", None)
        # chỉ patch trên Windows
        if os.name == "nt" and self._orig is not None:
            pathlib.PosixPath = pathlib.WindowsPath
        return self

    def __exit__(self, exc_type, exc, tb):
        if os.name == "nt" and getattr(self, "_orig", None) is not None:
            pathlib.PosixPath = self._orig


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

    def _mp_has_face(self, frame_bgr):
        print("Checking face presence with MediaPipe...")
        # MediaPipe FaceDetection cần RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_fd.process(rgb)
        return (res.detections is not None) and (len(res.detections) > 0)

    def load_model(self):
        repo_dir = Path(__file__).parent / "yolov5"
        weights = Path(__file__).parent / "models" / "yolo_based" / "yolo_v5.pt"
        print(f"Loading model from {weights}...")

        with _PosixPathPatcher():
            self.yolo_classifier = torch.hub.load(
                str(repo_dir),  # hoặc 'ultralytics/yolov5' nếu muốn tải từ GitHub
                "custom",
                path=str(weights),
                source="local",
                trust_repo=True,
                force_reload=True,  # ép bỏ cache cũ (tránh pickle lỗi do cache)
            )

            print("Class names:", self.yolo_classifier.names)

        m = self.yolo_classifier
        m.eval()
        # m.conf = 0.15         # ↓ giảm ngưỡng NMS confidence (v8 bạn dùng ~0.15)
        m.conf = 0.08
        # m.iou = 0.50
        m.iou = 0.45
        m.max_det = 20
        m.agnostic = False
        m.multi_label = False

        # Chuyển model về đúng device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(device)

        print("Model loaded to device:", device)

        # In names để kiểm tra mapping lớp
        print("NAMES:", m.names)

        # Nếu checkpoint có lớp 'background' thì loại ra khỏi infer
        if isinstance(m.names, (list, tuple)) and "background" in m.names:
            keep = [i for i, n in enumerate(m.names) if n != "background"]
            m.classes = keep
            print("Filtered classes (exclude 'background'):", keep)

    def init_variables(self):
        self.cap = cv2.VideoCapture(0)
        self.monitoring = False

        self.gif_frames = []
        self.gif_frame_index = 0
        self.is_gif_playing = True

        self.is_playing_warning = False
        self.last_phone_warning_time = 0
        self.warning_interval = 3

        self.last_boxes = []
        self.frame_count = 0
        self.frame_skip = 3
        self.start_time = time.time()

        self.yolo_predict_thread = None
        self.frame_for_predict = None
        self.predict_lock = threading.Lock()

        self.sleepy_eye_count = 0
        self.sleepy_yawn_count = 0
        self.look_away_count = 0
        self.phone_count = 0
        self.no_face_count = 0
        self.previous_statuses = ["natural"]

        self.using_phone = False

        # Trạng thái không phát hiện khuôn mặt
        self.start_no_face_time = None
        self.last_no_face_audio_time = None  # Thời điểm cuối cùng phát âm thanh cảnh báo không có khuôn mặt
        self.play_no_face_sound_step = 1.5  # giây
        self.max_no_face_duration = 5  # tối đa 5s nếu không phát hiện khuôn mặt thì sẽ phát ra loa dừng xe
        self.has_warned_stop = False

        # Trạng thái không nhìn thẳng
        self.start_look_away_time = None
        self.play_look_away_sound_step = 1.5  # giây

        # Khởi tạo thời gian gửi ảnh qua Telegram
        self.last_send_time = None
        self.send_interval = 10  # giây
        self.telegram_frame = None  # ảnh sẽ gửi qua Telegram

        # Khởi tạo về tình trạng buồn ngủ
        self.start_sleepy_eye_time = None
        self.maximum_sleepy_eye_duration = 10  # giây
        self.warned_3s = False
        self.warned_5s = False
        self.warned_10s = False
        self.is_playing_stop_warning = False  # Thông báo dừng xe có đang phát hay không

        self.mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0: gần (<=2m). Nếu cam xa hơn, thử 1
            min_detection_confidence=0.2,  # hạ ngưỡng một chút để tăng recall
        )

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

        self.no_face_label = tk.Label(
            stat_frame, text="🔍 Không phát hiện khuôn mặt: 0", font=("Helvetica", 12), fg="orange", anchor="w"
        )
        self.no_face_label.pack(fill="x", padx=20, pady=2)

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

    def post_process_predictions(self, labels):
        """
        Xử lý kết quả dự đoán từ mô hình phân loại hành vi lái xe.
        Bao gồm cập nhật trạng thái, âm thanh cảnh báo và thống kê.
        """
        # Loại bỏ nhãn 'natural'
        labels = [label for label in labels if label != "natural"]
        num_errors = len(labels)

        if num_errors == 0:
            self.reset_warnings()
            self.set_status("🟢 Tình trạng bình thường", "green")
            self.previous_statuses = ["natural"]
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

                # Kiểm tra xem có nên phát audio yêu cầu nhìn thẳng không?
                if (
                    self.start_look_away_time is None
                    or time.time() - self.start_look_away_time > self.play_look_away_sound_step
                ):
                    self.start_look_away_time = time.time()
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
                audio_path, is_warning_stop_car = self.play_sleepy_eye_by_level()

                if is_warning_stop_car:
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

    def refilter_classes(self):
        print(f"\n{'🔍 REFILTER_CLASSES':-^60}")
        print(f"📥 Input boxes: {len(self.last_boxes)}")

        final_classes = []
        final_last_boxes = []

        for idx, (x1, y1, x2, y2, conf, class_name) in enumerate(self.last_boxes, 1):
            print(f"\nBox #{idx}:")
            print(f"  ├─ Class: {class_name}")
            print(f"  ├─ Confidence: {conf:.4f} ({conf*100:.2f}%)")

            if class_name == "natural":
                print(f"  └─ ❌ REJECTED: Class is 'natural'")
                continue

            # Kiểm tra điều kiện lọc
            if class_name == "phone":
                if conf >= 0.6:
                    final_last_boxes.append((x1, y1, x2, y2, conf, class_name))
                    final_classes.append(class_name)
                    print(f"  └─ ✅ ACCEPTED: phone with conf {conf:.4f} >= 0.6")
                else:
                    print(f"  └─ ❌ REJECTED: phone confidence {conf:.4f} < 0.6")

            elif class_name in ["look_away", "rub_eye", "sleepy_eye", "yawn"]:
                thr = 0.30 if class_name == "look_away" else 0.38
                if conf >= thr:
                    final_last_boxes.append((x1, y1, x2, y2, conf, class_name))
                    final_classes.append(class_name)
                    print(f"  └─ ✅ ACCEPTED: {class_name} with conf {conf:.4f} >= {thr}")
                else:
                    print(f"  └─ ❌ REJECTED: {class_name} confidence {conf:.4f} < {thr}")
            else:
                print(f"  └─ ❌ REJECTED: Unknown class '{class_name}'")

        print(f"\n{'─'*60}")
        print(f"📤 Output: {len(final_classes)} classes accepted")
        print(f"   Classes: {final_classes}")
        print(f"{'─'*60}\n")

        return final_classes, final_last_boxes

    def run_predict(self):
        self.yolo_predict_thread = None

        with self.predict_lock:
            # results = self.yolo_classifier(self.frame_for_predict, size=512)
            results = self.yolo_classifier(self.frame_for_predict, size=640, augment=True)
            self.last_boxes = []
            result_classes = []

            # Log tổng số detections
            total_detections = len(results.xyxy[0])
            print(f"\n{'='*60}")
            print(f"🔍 TỔNG SỐ DETECTIONS: {total_detections}")

            if total_detections == 0:
                print("⚠️ KHÔNG CÓ DETECTION NÀO - Model không phát hiện được gì!")
                print(f"{'='*60}\n")
            else:
                print(f"{'='*60}")
                print(f"📊 CHI TIẾT CÁC DETECTIONS:")
                print(f"{'='*60}")

            for idx, (*xyxy, conf, cls) in enumerate(results.xyxy[0].tolist(), 1):
                x1, y1, x2, y2 = xyxy
                class_name = self.yolo_classifier.names[int(cls)]
                result_classes.append(class_name)

                # Log chi tiết từng detection
                print(f"Detection #{idx}:")
                print(f"  ├─ Class: {class_name}")
                print(f"  ├─ Confidence: {conf:.4f} ({conf*100:.2f}%)")
                print(f"  └─ BBox: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")

                if class_name == "natural":
                    print(f"  └─ ⏭️ Bỏ qua class 'natural'")
                    continue
                self.last_boxes.append((x1, y1, x2, y2, conf, class_name))

            print(f"{'='*60}")
            print(f"📋 Detected classes: {result_classes}")
            print(f"📦 Boxes được giữ lại (không phải 'natural'): {len(self.last_boxes)}")

            if not result_classes:
                # Fallback: nếu vẫn thấy mặt bằng MediaPipe thì đừng nói "không có mặt"
                try:
                    if self._mp_has_face(self.frame_for_predict):
                        # coi như "mất tập trung nhẹ" thay vì "no face"
                        self.set_status("🟠 Mất tập trung", "orange")
                        # (tuỳ bạn) tự tăng look_away_count nhẹ để thống kê
                        self.look_away_count += 1
                        self.look_label.config(text=f"👀 Nhìn hướng khác: {self.look_away_count}")
                        self.previous_statuses = ["look_away"]
                        now = time.time()
                        if (
                            self.start_look_away_time is None
                            or now - self.start_look_away_time > self.play_look_away_sound_step
                        ):
                            self.start_look_away_time = now
                            if not self.is_playing_stop_warning:
                                play_audio("./assets/audios/look_straight.wav")
                        return
                except Exception as _:
                    pass

                self.no_face_count += 1 if "no_face_detected" not in self.previous_statuses else 0
                self.no_face_label.config(text=f"🔍 Không phát hiện khuôn mặt: {self.no_face_count}")

                self.set_status("🔍 Không phát hiện khuôn mặt", "gray")
                self.reset_warnings()

                now = time.time()
                no_face_duration = now - (self.start_no_face_time or now)

                if self.start_no_face_time is None:
                    self.start_no_face_time = now

                if no_face_duration >= self.max_no_face_duration:
                    if not self.has_warned_stop:
                        self.has_warned_stop = True
                        self.is_playing_stop_warning = True

                        def play_stop_warning_then_reset():
                            pygame.mixer.music.load("./assets/audios/stop_car_warning.wav")
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                            self.is_playing_stop_warning = False

                        threading.Thread(target=play_stop_warning_then_reset, daemon=True).start()

                    elif (
                        not self.is_playing_warning and no_face_duration >= self.max_no_face_duration + 6
                    ):  # đã cảnh báo stop rồi, giờ tiếp tục cảnh báo cấp độ 3
                        self.is_playing_warning = True

                        def play_warn_level3():
                            play_audio("./assets/audios/warn_level3.wav")
                            self.is_playing_warning = False

                        threading.Thread(target=play_warn_level3, daemon=True).start()
                    elif not self.is_playing_warning and no_face_duration >= self.max_no_face_duration + 3:
                        self.is_playing_warning = True

                        def play_warn_level1():
                            play_audio("./assets/audios/warn_level1.wav")
                            self.is_playing_warning = False

                        threading.Thread(target=play_warn_level1, daemon=True).start()

                elif (
                    self.last_no_face_audio_time is None
                    or now - self.last_no_face_audio_time >= self.play_no_face_sound_step
                ):
                    play_audio("./assets/audios/no_face_detected.wav")
                    self.last_no_face_audio_time = now

                self.previous_statuses = ["no_face_detected"]
                return

            self.start_no_face_time = None  # Reset khi có khuôn mặt
            self.last_no_face_audio_time = None  # Reset khi có khuôn mặt
            self.has_warned_stop = False
            if "no_face_detected" in self.previous_statuses:
                self.previous_statuses.remove("no_face_detected")
            result_classes, result_last_boxes = self.refilter_classes()
            self.last_boxes = result_last_boxes
            print(f"Filtered classes: {result_classes}")

            if "phone" in result_classes:
                self.using_phone = True
                self.set_status("📱 Dùng điện thoại", "orange")
                self.phone_label.config(text=f"📱 Dùng điện thoại: {self.phone_count}")
                now = time.time()
                if now - self.last_phone_warning_time > self.warning_interval:
                    play_audio("./assets/audios/not_use_phone.wav")
                    self.phone_count += 1 if "phone" not in self.previous_statuses else 0
                    self.previous_statuses.append("phone")
                    self.last_phone_warning_time = now
            else:
                self.using_phone = False

            self.post_process_predictions(result_classes)

    def reset_warnings(self):
        self.start_sleepy_eye_time = None
        self.warned_3s = False
        self.warned_5s = False
        self.warned_10s = False

    def play_sleepy_eye_by_level(self):
        """
        Xác định âm thanh cảnh báo dựa trên khoảng thời gian mà mắt buồn ngủ đã được phát hiện.
        """
        if self.start_sleepy_eye_time is None:
            self.start_sleepy_eye_time = time.time()

        duration = time.time() - self.start_sleepy_eye_time
        audio_path = "./assets/audios/warn_level2.wav" if not self.warned_3s else "./assets/audios/warn_level3.wav"
        is_warning_stop_car = False  # flag xem có đang cảnh báo stop_car_warning

        # Cứ từng chu kỳ 6s sẽ gửi đến telegram
        now = time.time()
        if duration >= 6 and (self.last_send_time is None or now - self.last_send_time >= 6):
            self.last_send_time = now
            self.telegram_frame = self.frame_for_predict.copy()
            self.run_send_telegram_alert(duration)

        if duration >= 6 and not self.warned_5s:
            audio_path = "./assets/audios/stop_car_warning.wav"
            self.warned_5s = True
            is_warning_stop_car = True
        elif (duration >= 3 and not self.warned_3s) or (self.warned_3s and duration >= 7):
            audio_path = "./assets/audios/warn_level3.wav"
            self.warned_3s = True

        return audio_path, is_warning_stop_car

    def update_error_counts(self, labels):
        if "sleepy_eye" in labels:
            self.sleepy_eye_count += 1 if "sleepy_eye" not in self.previous_statuses else 0
            self.eye_label.config(text=f"👁️ Mắt buồn ngủ: {self.sleepy_eye_count}")
        if "yawn" in labels:
            self.sleepy_yawn_count += 1 if "yawn" not in self.previous_statuses else 0
            self.yawn_label.config(text=f"😪 Ngáp: {self.sleepy_yawn_count}")
        if "look_away" in labels:
            self.look_away_count += 1 if "look_away" not in self.previous_statuses else 0
            self.look_label.config(text=f"👀 Nhìn hướng khác: {self.look_away_count}")

        self.previous_statuses = ["natural"] if not labels else labels

    def run_send_telegram_alert(self, lasted_duration):
        """
        Gửi cảnh báo qua Telegram nếu tài xế nhắm mắt quá lâu.
        """
        if self.telegram_frame is not None:
            # Tạo thread gửi cảnh báo
            threading.Thread(
                target=send_driver_drowsiness_alert_from_frame,
                args=(self.telegram_frame, lasted_duration),
                daemon=True,
            ).start()

    def update_frame(self):
        if self.monitoring:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 400))
                current_time = time.time()

                if current_time - self.start_time >= 1:
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip == 0 and self.yolo_predict_thread is None:
                        self.frame_for_predict = frame.copy()
                        self.yolo_predict_thread = threading.Thread(target=self.run_predict)
                        self.yolo_predict_thread.start()
                        self.start_time = current_time

                color_map = {
                    "look_away": (255, 0, 0),  # đỏ
                    "natural": (128, 128, 128),  # xám
                    "phone": (0, 165, 255),  # cam
                    "rub_eye": (0, 255, 0),  # xanh lá
                    "sleepy_eye": (255, 255, 0),  # vàng
                    "yawn": (255, 0, 255),  # tím
                }

                for x1, y1, x2, y2, conf, class_name in self.last_boxes:
                    color = color_map.get(class_name, (255, 255, 255))  # trắng nếu ko có màu định sẵn
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_frame.configure(image=img)
                self.video_frame.image = img

            self.root.after(30, self.update_frame)

    def __del__(self):
        self.monitoring = False
        self.cap.release()
        try:
            if hasattr(self, "mp_fd") and self.mp_fd:
                self.mp_fd.close()
        except Exception:
            pass
        if self.yolo_predict_thread and self.yolo_predict_thread.is_alive():
            self.yolo_predict_thread.join(timeout=1)
        pygame.mixer.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.mainloop()
