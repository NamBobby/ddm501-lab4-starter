# DDM501.22 — Assignment 2: Credit Risk ML Pipeline

**Học viên:** Lê Thanh Phương Nam — **25MS23308**

**Môn học:** AI trong sản xuất: DevOps, DataOps, MLOps

## 1. Mục tiêu và phạm vi

Project xây dựng pipeline dự đoán khả năng credit default, kết hợp:

- Kiểm tra chất lượng dữ liệu.
- Preprocessing và feature engineering.
- So sánh 10 cấu hình mô hình.
- Tracking và model registry bằng MLflow.
- Orchestration bằng Apache Airflow.
- Quality gate trước khi đăng ký candidate.
- Đóng gói môi trường bằng Docker Compose.
- Kiểm tra prediction và xuất artifact staging.

Pipeline tiếp nối bài toán và kiến trúc của Assignment 1.

Phạm vi triển khai tự động hiện tại kết thúc ở staging bundle. Việc thay
model đang phục vụ FastAPI cần một bước phê duyệt và triển khai riêng.

## 2. Thiết kế pipeline

| Stage | Xử lý | Output |
|---|---|---|
| ingest | Snapshot dataset, config, Git commit và SHA-256 | Dataset và manifest |
| validate | Kiểm tra schema, thống kê và miền giá trị | Data validation report |
| prepare_features | Chia dữ liệu; fit preprocessing trên train | Split indices và fitted transformer |
| train | Huấn luyện classifier với cấu hình E01 | Candidate pipeline |
| evaluate | Đánh giá trên validation và theo nhóm | Validation metrics |
| quality_gate | Kiểm tra performance, fairness và group coverage | Gate report |
| register | Log model, signature, metrics và artifacts | Registered candidate URI |
| staging_smoke_test | So sánh MLflow/scikit-learn prediction | Smoke report và staging bundle |

Các task chạy tuần tự. Airflow dùng lịch `@monthly`, cho phép trigger
thủ công, `catchup=False` và tối đa một DAG run hoạt động.

Một số task có tối đa hai lần retry. Validation, quality gate,
registration và smoke test không tự retry. Failure callback ghi thông tin
lỗi và đường dẫn task log; chưa cấu hình gửi email hoặc Slack.

## 3. Dữ liệu và chống leakage

Dataset gồm 30.000 dòng và 23 input features, với target
`default_payment_next_month`. Script `scripts/make_dataset.py` cung cấp
dữ liệu theo schema credit-default dùng trong project.

Phân chia stratified, random seed **501**:

- Train: 18.000 dòng — 60%.
- Validation: 6.000 dòng — 20%.
- Test: 6.000 dòng — 20%.

Imputation, scaling và encoding chỉ được fit trên train.
Feature engineering bổ sung các tỷ lệ sử dụng hạn mức, thanh toán và
thống kê lịch sử chậm trả.

Trong experiment sweep, mô hình được chọn bằng validation PR AUC.
Test set dùng để đánh giá mô hình đã chọn. Pipeline Airflow sử dụng
validation cho quality gate và không đánh giá lại test set.

## 4. Kết quả experiment sweep

| ID | Model | Engineered features | Validation PR AUC | Validation ROC AUC |
|---|---|---|---:|---:|
| E01 | Logistic Regression | Có | 0.5514 | 0.7519 |
| E02 | Logistic Regression | Có | 0.5512 | 0.7519 |
| E03 | Logistic Regression | Có | 0.5512 | 0.7519 |
| E04 | Random Forest | Có | 0.5411 | 0.7481 |
| E05 | Random Forest | Có | 0.5400 | 0.7450 |
| E06 | Random Forest | Có | 0.5407 | 0.7462 |
| E07 | HistGradientBoosting | Có | 0.5422 | 0.7450 |
| E08 | HistGradientBoosting | Có | 0.5422 | 0.7450 |
| E09 | HistGradientBoosting | Có | 0.5410 | 0.7446 |
| E10 | HistGradientBoosting | Không | 0.5340 | 0.7408 |

**Selected configuration:** E01, Logistic Regression, `C=0.1`,
`max_iter=2000`, engineered features.

E01 đạt validation PR AUC cao nhất trong các cấu hình đã thử.
Chênh lệch với E02/E03 rất nhỏ; kết quả một lần split chưa chứng minh
ưu thế có ý nghĩa thống kê.

E08 và E10 dùng cùng cấu hình HGB nhưng khác feature engineering:
PR AUC tăng khoảng 0.0081 khi bổ sung derived features.

Kết quả test của E01:

| Metric | Value |
|---|---:|
| ROC AUC | 0.7506 |
| PR AUC | 0.5534 |
| Precision, threshold 0.30 | 0.5302 |
| Recall, threshold 0.30 | 0.5089 |
| Brier score | 0.1441 |

Chi tiết cấu hình: `config/experiment.yaml`.
Kết quả đầy đủ: `artifacts/assignment2/results.csv`.

## 5. Quality gates

Cấu hình tại `config/orchestration.yaml`:

| Điều kiện | Ngưỡng |
|---|---:|
| Validation ROC AUC | >= 0.70 |
| Validation PR AUC | >= 0.45 |
| Fairness gap | <= 0.10 |
| Số nhóm đủ điều kiện đánh giá | >= 2 |

Fairness gap đo chênh lệch tỷ lệ hồ sơ được flag ở threshold 0.30
giữa các nhóm SEX đủ điều kiện đánh giá. Đây là tín hiệu kiểm tra,
không phải bằng chứng đầy đủ về tính công bằng.

Run local đã đạt ROC AUC 0.7519, PR AUC 0.5514 và fairness gap khoảng 0.031.
Run manual qua Airflow đã hoàn thành cả 8 task.

Minh chứng Docker được lưu tại `docs/evidence/orchestrated-run/`.

## 6. Chạy Airflow và MLflow bằng Docker

Yêu cầu: Git, Docker Desktop đang chạy Linux containers và Docker Compose.

Clone đúng nhánh:

```powershell
git clone --branch assignment2 --single-branch https://github.com/NamBobby/ddm501-lab4-starter.git ddm501-assignment2
Set-Location ".\ddm501-assignment2"
```

Build image trước, sau đó khởi động hai service:

```powershell
$env:ASSIGNMENT2_GIT_COMMIT = (git rev-parse HEAD).Trim()
docker compose -f compose.assignment2.yaml build airflow
docker compose -f compose.assignment2.yaml up -d
docker compose -f compose.assignment2.yaml logs --tail 60 airflow
```

Lần đầu cần chờ Airflow khởi tạo và nhận diện DAG.

| Service | URL |
|---|---|
| Airflow | http://localhost:18080 |
| MLflow của orchestration | http://localhost:5001 |

Airflow username: `admin`. Lấy password:

```powershell
docker compose -f compose.assignment2.yaml exec airflow cat /opt/airflow/standalone_admin_password.txt
```

Kiểm tra import và bật DAG:

```powershell
docker compose -f compose.assignment2.yaml exec airflow airflow dags list-import-errors
docker compose -f compose.assignment2.yaml exec airflow airflow dags unpause assignment2_credit_risk
```

Nếu chưa thấy DAG, đợi thêm rồi chạy lại. Khi DAG bị pause, manual run
có thể nằm trong queue và các task chưa được scheduler thực thi.

Trigger một run và kiểm tra sau khi scheduler xử lý:

```powershell
$runId = "demo-" + (Get-Date -Format "yyyyMMdd-HHmmss")
docker compose -f compose.assignment2.yaml exec airflow airflow dags trigger assignment2_credit_risk --run-id $runId
docker compose -f compose.assignment2.yaml exec airflow airflow tasks states-for-dag-run assignment2_credit_risk $runId
```

Kết quả mong đợi: cả 8 task `success`.
Trong MLflow, mở experiment `assignment2-orchestrated`.

Sau khi unpause, lịch monthly cũng có thể tạo một scheduled run.
Mỗi run có artifacts riêng và có thể tạo một candidate version mới.

Dừng services và giữ dữ liệu:

```powershell
docker compose -f compose.assignment2.yaml down
```

## 7. Chạy lại experiment sweep trên Windows

Yêu cầu Python 3.11:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-assignment2.txt
& .\.venv\Scripts\python.exe -m scripts.run_experiments
& .\.venv\Scripts\python.exe -m scripts.verify_candidate
```

MLflow experiment sweep sử dụng `mlflow.db` tại project root:

```powershell
& .\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Không nhầm ba tracking stores:

| Môi trường | Tracking store |
|---|---|
| Experiment sweep trên Windows | `mlflow.db` tại project root |
| Chạy pipeline stages trên Windows | `artifacts/orchestration/mlflow.db` |
| Airflow trong Docker | `/opt/runtime/mlflow.db` trong Docker volume |

Model version chỉ có ý nghĩa trong registry tương ứng.
Không dùng database có artifact paths Windows làm tracking store trong Docker.

## 8. Reproducibility và versioning

- Code và configuration được quản lý trên nhánh `assignment2`.
- Manifest ghi Git commit và SHA-256 của dataset snapshot.
- Snapshot được kiểm tra hash trước khi đọc tại các stage dùng dữ liệu.
- Split indices được lưu riêng cho từng run.
- Dependencies chính được pin trong requirements và Docker image.
- Airflow và ML dependencies sử dụng hai Python environments riêng.
- MLflow lưu parameters, metrics, model signature và artifacts.
- Probability output của pyfunc là ma trận hai cột qua `predict_proba`.
- Smoke test xác nhận output shape và prediction khớp scikit-learn.

Runtime database và artifacts lớn không được đưa lên Git.
Các báo cáo JSON minh chứng được lưu trong `docs/evidence/`.

## 9. Giới hạn và hướng mở rộng

Môi trường minh chứng dùng SQLite và SequentialExecutor.
Để mở rộng cần PostgreSQL, executor phù hợp, shared artifact storage,
quản lý secrets và kênh thông báo lỗi.

Pipeline xuất staging bundle nhưng chưa thực hiện promotion,
canary deployment hoặc rollback tự động cho API.

Lịch monthly hiện đọc dataset snapshot trong image. Khi có dữ liệu mới,
cần cập nhật dữ liệu và rebuild image; kết nối nguồn dữ liệu cập nhật
tự động là bước mở rộng.

## 10. Các file chính

| File | Vai trò |
|---|---|
| `config/experiment.yaml` | 10 cấu hình thực nghiệm |
| `config/orchestration.yaml` | Candidate selection và quality gates |
| `scripts/run_experiments.py` | Experiment sweep |
| `scripts/verify_candidate.py` | Kiểm tra model signature và artifact |
| `scripts/pipeline_stages.py` | Các stage của pipeline |
| `dags/assignment2_credit_risk.py` | Airflow DAG |
| `Dockerfile.airflow` | Airflow image và ML environment |
| `compose.assignment2.yaml` | Airflow và MLflow services |
| `artifacts/assignment2/` | Kết quả experiment sweep |
| `docs/evidence/orchestrated-run/` | Minh chứng chạy qua scheduler |

FastAPI và monitoring stack được giữ trong `app/`, `monitoring/`
và `docker-compose.yml`. Stack này dùng model tại `models/`;
candidate từ Airflow chưa tự động thay thế model đó.
