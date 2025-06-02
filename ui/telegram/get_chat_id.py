import requests

BOT_TOKEN = '7946972043:AAElgtTBs9KM0Uuu6zA47ld2wIWU5n-L1Z4'
url = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
r = requests.get(url)
print(r.json())