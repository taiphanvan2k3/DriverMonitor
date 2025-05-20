import tkinter as tk
from tkinter import ttk
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import random
import threading
import pygame

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

        # Webcam frame
        self.video_frame = tk.Label(root, bg="black")
        self.video_frame.place(x=20, y=20, width=540, height=400)

        # Status panel
        status_frame = tk.LabelFrame(root, text="Trạng thái tài xế", font=("Helvetica", 14, "bold"), fg="#333")
        status_frame.place(x=580, y=20, width=320, height=100)

        self.status_label = tk.Label(status_frame, text="🟢 Bình thường", font=("Helvetica", 16), fg="green")
        self.status_label.pack(pady=20)

        # Statistics panel
        stat_frame = tk.LabelFrame(root, text="📊 Thống kê cảnh báo", font=("Helvetica", 14))
        stat_frame.place(x=580, y=140, width=320, height=400)

        # --- Buồn ngủ ---
        tk.Label(stat_frame, text="🔴 Buồn ngủ", font=("Helvetica", 13, "bold"), fg="red").pack(
            anchor="w", padx=10, pady=(5, 0)
        )
        self.sleepy_eye_count = 0
        self.sleepy_yawn_count = 0

        self.eye_label = tk.Label(
            stat_frame, text="👁️ Mắt buồn ngủ: 0 ", font=("Segoe UI Emoji", 12), fg="red", anchor="w", justify="left"
        )

        self.eye_label.pack(fill="x", padx=20, pady=2)

        self.yawn_label = tk.Label(stat_frame, text="😪 Ngáp: 0", font=("Segoe UI Emoji", 12), fg="red", anchor="w")
        self.yawn_label.pack(fill="x", padx=20, pady=2)

        # --- Mất tập trung ---
        tk.Label(stat_frame, text="🟠 Mất tập trung", font=("Helvetica", 13, "bold"), fg="orange").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.lookaway_count = 0
        self.phone_count = 0

        self.look_label = tk.Label(
            stat_frame, text="👀 Nhìn hướng khác: 0", font=("Helvetica", 12), fg="orange", anchor="w"
        )
        self.look_label.pack(fill="x", padx=20, pady=2)

        self.phone_label = tk.Label(
            stat_frame, text="📱 Dùng điện thoại: 0", font=("Helvetica", 12), fg="orange", anchor="w"
        )
        self.phone_label.pack(fill="x", padx=20, pady=2)

        # Buttons
        self.start_button = ttk.Button(root, text="Start", command=self.start_monitoring)
        self.start_button.place(x=600, y=580)

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_monitoring, state="disabled")
        self.stop_button.place(x=680, y=580)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.monitoring = False

    def update_frame(self):
        if self.monitoring:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 400))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.video_frame.configure(image=img)
                self.video_frame.image = img

            # Simulate detection
            r = random.random()
            if r < 0.01:
                self.set_status("👁️ Mắt buồn ngủ", "red")
                self.sleepy_eye_count += 1
                self.eye_label.config(text=f"👁️ Mắt buồn ngủ: {self.sleepy_eye_count}")
                play_audio("audios/warn_level2.wav")

            elif r < 0.02:
                self.set_status("😪 Ngáp", "red")
                self.sleepy_yawn_count += 1
                self.yawn_label.config(text=f"😪 Ngáp: {self.sleepy_yawn_count}")
                play_audio("audios/warn_level3.wav")

            elif r < 0.03:
                self.set_status("👀 Nhìn hướng khác", "orange")
                self.lookaway_count += 1
                self.look_label.config(text=f"👀 Nhìn hướng khác: {self.lookaway_count}")
                play_audio("audios/warn_level1.wav")

            elif r < 0.035:
                self.set_status("📱 Dùng điện thoại", "orange")
                self.phone_count += 1
                self.phone_label.config(text=f"📱 Dùng điện thoại: {self.phone_count}")
                play_audio("audios/warn_level1.wav")

            else:
                self.set_status("🟢 Bình thường", "green")

            self.root.after(30, self.update_frame)

    def start_monitoring(self):
        self.monitoring = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.update_frame()

    def stop_monitoring(self):
        self.monitoring = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def set_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def __del__(self):
        self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = DriverMonitorApp(root)
    root.mainloop()
