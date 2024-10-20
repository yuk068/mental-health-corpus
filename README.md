# Phát hiện ngôn ngữ độc hại trong chủ đề đánh giá sức khỏe tâm lý

### Thành viên:
- Phạm Đức Duy - 23001855 - MAT3533 1
- Trần Hoàng Đạt - 22001558 - MAT3533 5-6-7 

### Dữ liệu: [Mental Health Corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus)

### Mô tả bài toán:

Bộ dữ liệu văn bản có 27977 mẫu văn bản đã được xử lý 1 phần, đầu ra là một nhãn số nguyên. Với cột đầu ra, nhãn 1 đại diện cho văn bản được coi là độc hại và có thể biểu thị cho nguy cơ về những vấn đề tâm lý, trong khi nhãn 0 được coi là văn bản có thể bỏ qua trong chủ đề được nói trên. Chúng ta sẽ khám phá những phương pháp áp dụng học máy khi làm việc với 1 bộ dữ liệu corpus (1 tập hợp các văn bản), thực hiện giảm chiều, phân cụm, trực quan hóa, và cuối cùng xây dựng những mô hình phân loại khác nhau để giải bài toán trên.

### Mô hình:
- Tiền xử lý văn bản:
  - TF-IDF
  - Word2Vec
- Giảm chiều dữ liệu:
  - PCA
  - LDA
- Phân cụm dữ liệu:
  - K - Means
  - GMM
- Phân loại dữ liệu:
  - Logistic Regression
  - SVM
  - MLP
  - RNN

### Reference Drive: [Mental Health Corpus Reference](https://drive.google.com/drive/u/1/folders/11GuJep29z7McalT862tMo4iGPzP-EAg3)

# Tiến độ dự án

### 20/10/2024
- (yuk) Đưa ra 1 bản corpus đã clean mẫu, sẽ dựa vào quy trình tạo ra bộ corpus này để thực hiện 1 notebook formal
- (yuk) Sẽ thực hiện các notebook formal để phân tích bộ corpus raw, cleaned, và so sánh chúng
- (yuk) Gợi ý: cần thực hiện trước các thí nghiệm về các sự kết hợp sau:
  - Vector hóa: TF - IDF, W2V (Skip-gram)
  - Giảm chiều: PCA, LDA (nhớ standardize trước khi fit và đặc biệt tránh data leakage (`fit_transfrom` trên bộ train, `transform` trên bộ test))
  - Phân cụm: K-means, GMM (chưa tìm hiểu)
  - Phân loại: Logistic Regression, SVM, MLP
  - Ngoài ra cần tìm hiểu thêm về Regularization, cách chọn parameter cho các model Vector hóa, từ các thí nghiệm đã thực hiện cho thấy các classifier rất dễ overfit
