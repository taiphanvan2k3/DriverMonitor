import requests
from datetime import datetime

BOT_TOKEN = '7946972043:AAElgtTBs9KM0Uuu6zA47ld2wIWU5n-L1Z4'
CHAT_ID = '8003237973'
PHOTO_PATH = 'D:\\School\\4thYear\\2ndSemester\\ComputerVision\\LearningMonitor\\ui\\image.png'

# Lấy ngày giờ hiện tại
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Ghi thông tin cảnh báo
status = '⚠️ Cảnh báo: Tài xế có dấu hiệu buồn ngủ'
caption = f'{status}\n🕒 Thời gian: {timestamp}'

url = f'https://shy-dream-c718.taiphanvan2403.workers.dev/bot{BOT_TOKEN}/sendPhoto'

with open(PHOTO_PATH, 'rb') as photo:
    response = requests.post(url, data={'chat_id': CHAT_ID, 'caption': caption}, files={'photo': photo})

print(response.json())