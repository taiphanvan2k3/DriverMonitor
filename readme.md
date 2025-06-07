# 🚗 Driver Drowsiness Detection

## 📌 Giới thiệu

Dự án này nhằm phát hiện trạng thái buồn ngủ của tài xế bằng cách sử dụng thị giác máy tính. Hệ thống có thể nhận diện các dấu hiệu như ngáp, nhắm mắt lâu, hoặc quay mặt đi nhằm đưa ra cảnh báo sớm, góp phần giảm tai nạn giao thông.

## 🧪 Các phương pháp thử nghiệm

Trong quá trình phát triển, tôi đã thử nghiệm 3 phương pháp khác nhau:

1. **CNN cơ bản**  
   - Mô hình mạng nơ-ron tích chập được xây dựng từ đầu bằng Keras.
2. **MobileNetV2**  
   - Mô hình nhẹ, được tối ưu cho thiết bị di động và nhúng.
3. **YOLO (You Only Look Once)**  
   - Dùng để phát hiện các biểu hiện mệt mỏi (như ngáp, nhắm mắt) trên khuôn mặt tài xế.

## 📂 Dữ liệu

- Dữ liệu được thu thập từ các bộ dataset công khai và từ các ảnh trích ra từ video do nhóm thực hiện.
Link Dataset: [Driver Drowsiness Detection Dataset](https://www.kaggle.com/datasets/ashishpatel26/driver-drowsiness-detection-dataset)
- Kết hợp với Mediapipe Face để trích xuất vùng mặt cần phân tích.

### Cho mô hình CNN-based (CNN cơ bản, MobileNetV2)
- **Dataset**: `dataset_multi_label_final`
- **Đặc điểm**: Gồm các ảnh khuôn mặt đã gán nhãn đa nhãn (multi-label) với các trạng thái như: natural (tỉnh táo), yawn (ngáp), sleepy_eye (nhắm mắt), look_away (quay mặt), rub_eye (dụi mắt)
- Link dataset: [Multi-label Face Dataset](https://drive.google.com/drive/u/1/folders/1MudVQ0-3eeB83H0-0QKuZPV269bh5du6?usp=sharing)

### Cho mô hình YOLOv11 (fine-tune)
- **Dataset**: `Dataset fine-tune Yolo`
- **Đặc điểm**: Dữ liệu dạng ảnh với bounding box cho các hành vi cần nhận diện (ngáp, nhắm mắt, quay mặt...) theo định dạng YOLO.
- Link dataset: [YOLOv11 Dataset](https://app.roboflow.com/dut-learning/driver-drowsiness-v2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## ⚙️ Kỹ thuật sử dụng

- **Tiền xử lý ảnh**: resize, grayscale, face crop.
- **Huấn luyện mô hình**: sử dụng TensorFlow/Keras cho CNN & MobileNetV2, PyTorch cho YOLO.
- **Phát hiện thời gian thực**: kết hợp OpenCV để nhận diện webcam.

## 📊 So sánh kết quả chi tiết

| Tiêu chí                             | CNN cơ bản | MobileNetV2 | YOLOv11s |
|-------------------------------------|------------|-------------|-----------|
| **Accuracy (%)**                    | 21.15      | 85.26       | 94.87     |
| **F1 Score Macro (%)**              | 12.56      | 81.55       | 95.85     |
| **F1 Score Weighted (%)**           | 7.70       | 81.99       | 95.03     |
| **Thời gian trung bình mỗi ảnh (s)**| 0.1037     | 0.1083      | 0.0467    |
| **Tổng thời gian xử lý 156 ảnh (s)**| 16.1755    | 16.8899     | 7.2826    |
| **Cần thư viện phụ trợ (Mediapipe)**| Có         | Có          | Không     |

## ✅ Kết luận

- **CNN cơ bản** phù hợp để hiểu cách hoạt động của mô hình từ đầu, nhưng hiệu quả thực tế còn hạn chế.
- **MobileNetV2** mang lại độ chính xác cao với tốc độ xử lý nhanh, phù hợp cho thiết bị tài nguyên thấp.
- **YOLOv11s** cho kết quả vượt trội cả về độ chính xác lẫn tốc độ, đồng thời không cần thư viện phụ trợ như Mediapipe.

➡️ **Vì vậy, tôi chọn mô hình YOLOv11s để triển khai vào ứng dụng phát hiện mệt mỏi tài xế trong thời gian thực.**

## 🖥️ Xây dựng ứng dụng

Ứng dụng phát hiện mệt mỏi tài xế được xây dựng dưới dạng giao diện người dùng (UI) bằng **Tkinter**, kết hợp với mô hình **YOLOv11s** đã fine-tune để phát hiện các hành vi như **ngáp**, **nhắm mắt lâu**, **quay mặt**, v.v.

### 🔧 Công nghệ sử dụng

- **Tkinter**: tạo giao diện đồ họa đơn giản và dễ triển khai.
- **YOLOv11s** (PyTorch): phát hiện hành vi mệt mỏi trên khuôn mặt tài xế trong thời gian thực.
- **OpenCV**: đọc hình ảnh từ webcam và hiển thị kết quả phân tích.
- **PyTorch**: dùng để load mô hình đã huấn luyện và thực hiện suy luận.

### ⚙️ Tính năng chính của ứng dụng

- Giao diện trực quan gồm các nút: *Start Detection*, *Stop*, *Exit*.
- Hiển thị video từ webcam và vẽ **bounding box** với **nhãn hành vi**.
- Tự động phát cảnh báo hoặc gửi thông báo đến người thân ở Telegram nếu phát hiện hành vi nguy hiểm (ví dụ: phát hiện ngáp nhiều lần hoặc nhắm mắt kéo dài).
- Không yêu cầu Mediapipe – YOLO hoạt động độc lập với độ chính xác cao.

- **Sơ đồ hệ thống**:

  - 📌 *Sơ đồ Usecase tổng quan hệ thống*  
    ![image](https://github.com/user-attachments/assets/f293ba35-f5f7-4507-a039-f62f0557ed0f)

  - 🔁 *Sơ đồ tuần tự của quá trình phát hiện hành vi người dùng*  
    ![image](https://github.com/user-attachments/assets/787d17fd-4038-42da-9c7c-1433f0f5c9f4)

  - 📤 *Sơ đồ luồng mô tả quy trình phát cảnh báo và gửi thông báo đến người thân*  
    ![image](https://github.com/user-attachments/assets/bfd5ac1e-f9f2-43b3-90cb-9d60ba62dd53)

### 📸 Giao diện mẫu
![image](https://github.com/user-attachments/assets/54777ad5-27da-41f1-b809-3c9017f8dc6e)
![image](https://github.com/user-attachments/assets/f7f566f4-3fa0-4982-9a4e-de0ffae7cfb0)
![image](https://github.com/user-attachments/assets/1b1ec1af-2d86-48ee-82b4-d97587368dee)
![image](https://github.com/user-attachments/assets/5e626558-ffcd-4607-881f-2952f618e3a9)
![image](https://github.com/user-attachments/assets/8a31f1fd-2bd0-43c7-b63b-03ba2b64d410)




