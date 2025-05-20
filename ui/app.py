import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageSequence
import threading
import pygame
from ultralytics import YOLO
import time


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

    def init_variables(self):
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
        tk.Label(stat_frame, text="🔴 Buồn ngủ", font=("Helvetica", 13, "bold"), fg="red").pack(anchor="w", padx=10, pady=(5, 0))
        self.eye_label = tk.Label(stat_frame, text="👁️ Mắt buồn ngủ: 0", font=("Segoe UI Emoji", 12), fg="red", anchor="w")
        self.eye_label.pack(fill="x", padx=20, pady=2)

        self.yawn_label = tk.Label(stat_frame, text="😪 Ngáp: 0", font=("Segoe UI Emoji", 12), fg="red", anchor="w")
        self.yawn_label.pack(fill="x", padx=20, pady=2)

        # Mất tập trung
        tk.Label(stat_frame, text="🟠 Mất tập trung", font=("Helvetica", 13, "bold"), fg="orange").pack(anchor="w", padx=10, pady=(10, 0))
        self.look_label = tk.Label(stat_frame, text="👀 Nhìn hướng khác: 0", font=("Helvetica", 12), fg="orange", anchor="w")
        self.look_label.pack(fill="x", padx=20, pady=2)

        self.phone_label = tk.Label(stat_frame, text="📱 Dùng điện thoại: 0", font=("Helvetica", 12), fg="orange", anchor="w")
        self.phone_label.pack(fill="x", padx=20, pady=2)

    def init_buttons(self):
        self.start_icon = ctk.CTkImage(Image.open("./assets/start.png").resize((24, 24)))
        self.stop_icon = ctk.CTkImage(Image.open("./assets/stop.png").resize((24, 24)))

        self.start_button = ctk.CTkButton(
            master=self.root, text="Start", image=self.start_icon, compound="left",
            command=self.start_monitoring, fg_color="#4CAF50", hover_color="#45A049",
            text_color="white", corner_radius=10, font=("Arial", 12), width=100, height=36
        )
        self.start_button.place(x=600, y=580)

        self.stop_button = ctk.CTkButton(
            master=self.root, text="Stop", image=self.stop_icon, compound="left",
            command=self.stop_monitoring, fg_color="#f44336", hover_color="#d32f2f",
            text_color="white", corner_radius=10, font=("Arial", 12), width=100, height=36
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

    def predict_yolo(self, frame):
        with self.predict_lock:
            results = self.yolo_model.predict(frame, conf=0.6, classes=[67], verbose=False)
            self.last_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                self.last_boxes.append((x1, y1, x2, y2, conf))

            if self.last_boxes:
                self.set_status("📱 Dùng điện thoại", "orange")
                self.phone_label.config(text=f"📱 Dùng điện thoại: {self.phone_count}")
                now = time.time()
                if now - self.last_warning_time > self.warning_interval:
                    self.phone_count += 1
                    play_audio("./assets/audios/warn_level1.wav")
                    self.last_warning_time = now
            else:
                self.set_status("🟢 Bình thường", "green")
                self.last_warning_time = 0

    def run_predict_thread(self):
        self.predict_yolo(self.frame_for_predict)
        self.predict_thread = None

    def update_frame(self):
        if self.monitoring:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 400))

                if time.time() - self.start_time >= 3:
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip == 0 and self.predict_thread is None:
                        self.frame_for_predict = frame.copy()
                        self.predict_thread = threading.Thread(target=self.run_predict_thread)
                        self.predict_thread.start()

                for x1, y1, x2, y2, conf in self.last_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Cell phone {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
