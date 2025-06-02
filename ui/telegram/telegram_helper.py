import cv2
import requests
from datetime import datetime
from io import BytesIO

BOT_TOKEN = "7946972043:AAElgtTBs9KM0Uuu6zA47ld2wIWU5n-L1Z4"
CHAT_ID = "8003237973"


def send_driver_drowsiness_alert_from_frame(frame, lasted_duration):
    """
    Gửi cảnh báo buồn ngủ của tài xế qua Telegram.
    frame: Frame từ video chứa hình ảnh của tài xế
    lasted_duration: Thời gian tài xế đã có dấu hiệu buồn ngủ (tính bằng giây)
    """
    # Lấy ngày giờ hiện tại
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "⚠️ Cảnh báo: Tài xế có dấu hiệu buồn ngủ"
    caption = f"{status}\n🕒 Thời gian: {timestamp}\n⏳ Thời gian buồn ngủ: {int(lasted_duration)} giây"

    # Mã hóa frame thành ảnh JPEG
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Không thể mã hóa ảnh từ frame")

    # Tạo file-like object từ ảnh đã mã hóa
    image_io = BytesIO(encoded_image.tobytes())
    image_io.name = "alert.jpg"  # Telegram yêu cầu tên file

    # Gửi ảnh qua Telegram
    url = f"https://shy-dream-c718.taiphanvan2403.workers.dev/bot{BOT_TOKEN}/sendPhoto"
    response = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": image_io})

    return response.json()
