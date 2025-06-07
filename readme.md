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
- **Đặc điểm**: Gồm các ảnh khuôn mặt đã gán nhãn đa nhãn (multi-label) với các trạng thái như: tỉnh táo, ngáp, nhắm mắt, quay mặt, mệt mỏi...
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
