# Phát hiện ngôn ngữ độc hại trong chủ đề đánh giá sức khỏe tâm lý

### Thành viên:
- Phạm Đức Duy - 23001855 - MAT3533 1
- Trần Hoàng Đạt - 22001558 - MAT3533 5-6-7 

### Dữ liệu: [Mental Health Corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus)

### Mô tả bài toán:

Bộ dữ liệu văn bản có 27977 mẫu văn bản đã được xử lý 1 phần, đầu ra là một nhãn số nguyên. Với cột đầu ra, nhãn 1 đại diện cho văn bản được coi là độc hại và có thể biểu thị cho nguy cơ về những vấn đề tâm lý, trong khi nhãn 0 được coi là văn bản có thể bỏ qua trong chủ đề được nói trên. Chúng ta sẽ khám phá những phương pháp phân tích dữ liệu và áp dụng học máy khi làm việc với 1 bộ dữ liệu Corpus (1 tập hợp các văn bản), thực hiện trực quan hóa (có liên quan tới giảm chiều) phân cụm, và cuối cùng xây dựng những mô hình phân loại khác nhau để giải bài toán trên.

### Nội dung nghiên cứu, báo cáo:
- Trích xuất đặc trưng từ văn bản:
  - TF-IDF (Term Frequency - Inverse Document Frequency)
- Trực quan hóa:
  - LSA (Latent Semantic Analysis) - 1 Kỹ thuật phân tích / giảm chiều
- Phân cụm dữ liệu:
  - K - Means
  - GMM (Gaussian Mixture Model)
- Phân loại dữ liệu:
  - KNN (K-Nearest Neighbors)
  - Random Forest
  - Logistic Regression
  - SVM (Support Vector Machine)


### Reference Drive: [Mental Health Corpus Reference](https://drive.google.com/drive/u/1/folders/11GuJep29z7McalT862tMo4iGPzP-EAg3)

# Tiến độ dự án

### 20/10/2024
- (yuk) Đưa ra 1 bản corpus đã clean mẫu, sẽ dựa vào quy trình tạo ra bộ corpus này để thực hiện 1 notebook formal.
- (yuk) Sẽ thực hiện các notebook formal để phân tích bộ corpus raw, cleaned, và so sánh chúng.
- (yuk) Gợi ý: cần thực hiện trước các thí nghiệm về các sự kết hợp sau:
  - Vector hóa: TF - IDF, W2V (Skip-gram).
  - Giảm chiều: PCA, LDA (nhớ standardize trước khi fit và đặc biệt tránh data leakage (`fit_transfrom` trên bộ train, `transform` trên bộ test)).
  - Phân cụm: K-means, GMM (chưa tìm hiểu).
  - Phân loại: Logistic Regression, SVM, MLP.
  - Ngoài ra cần tìm hiểu thêm về Regularization, cách chọn parameter cho các model Vector hóa, từ các thí nghiệm đã thực hiện cho thấy các classifier rất dễ overfit.
 
### 22/10/2024
- (yuk) Đưa ra 1 notebook chính thức phân tích bộ dữ liệu gốc, sẽ dựa vào để dọn dữ liệu và đưa ra 1 bộ dữ liệu mới.
- (yuk) Sau khi đưa ra 1 bộ dữ liệu mới, sẽ so sánh bộ cleaned với bộ raw.

### 25/10/2024
- (yuk) Thực hiện lại 4 Notebook chính thức prefixed từ 1-4. Đưa ra bộ dữ liệu đã dọn mới.
- (yuk) Chốt lại về nội dung nghiên cứu, báo cáo.
- (yuk) Sẽ thực hiện thêm các Notebook thực hiện tất cả các nội dung còn lại.
- (yuk) Sẽ đưa ra các phỏng đoán và nhận xét, cần được kiểm chứng với các nguồn đáng tin cậy.

### 01/11/2024
- (yuk) Bổ sung các Density 2D Plot tại `./plots/2D`