# Ứng dụng các phương pháp học máy trong chủ đề đánh giá sức khỏe tâm lý

### [Github Repo: Mental health corpus](https://github.com/yuk068/mental-health-corpus)

### Thành viên:
- Phạm Đức Duy - 23001855 - MAT3533 1
- Trần Hoàng Đạt - 22001558 - MAT3533 5-6-7 

### Dữ liệu: [Mental Health Corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus)

### Mô tả bài toán:

Bộ dữ liệu văn bản có 27977 mẫu văn bản đã được xử lý 1 phần, đầu ra là một nhãn số nguyên. Với cột đầu ra, nhãn 1 đại diện cho văn bản được coi là độc hại và có thể biểu thị cho nguy cơ về những vấn đề tâm lý, trong khi nhãn 0 được coi là văn bản có thể bỏ qua trong chủ đề được nói trên. Chúng ta sẽ khám phá những phương pháp phân tích dữ liệu và áp dụng học máy khi làm việc với 1 bộ dữ liệu Corpus (1 tập hợp các văn bản), thực hiện trực quan hóa phân cụm, và cuối cùng xây dựng những mô hình phân loại khác nhau để giải bài toán trên.

### Nội dung nghiên cứu, báo cáo:
- Trích xuất đặc trưng từ văn bản:
	- TF-IDF (Term Frequency - Inverse Document Frequency)
- Trực quan hóa:
	- LSA (Latent Semantic Analysis)
- Phân cụm dữ liệu:
	- K - Means
	- GMM (Gaussian Mixture Model)
- Phân loại dữ liệu:
	- KNN (K-Nearest Neighbors)
	- Random Forest
	- Logistic Regression

### Bổ sung cho cuối kỳ:

- Phân loại dữ liệu:
	- Naive Bayes
	- SVM (Support Vector Machine)
  
- Phân tích hồi quy: Trích xuất biến hồi quy sử dụng hàm quyết định của mô hình Logistic Regression tốt nhất.
	- KNN (K-Nearest Neighbors) Regression
	- Random Forest Regression

### Cấu trúc mã nguồn

- Giữa kì: Chạy theo thứ tự các Jupyter Notebook `.ipynb` từ A đến K.
- Cuối kì:
	- 2 chương phân loại mới chạy trong file `V_naive_bayes.ipynb` và `X_svm.ipynb`.
	- Trích xuất bộ dữ liệu hồi quy ở file `Y_extract_regression_dataset.ipynb`.
 	- Thực hiện hồi quy cho cả 2 mô hình Linear Regression và Random Forest trong file `Z_regression.ipynb`.
