# DDM501.22 — Lab 4: Monitoring & Production Deployment

## Nhóm 7

| Thành viên | Mã học viên |
|---|---|
| Dương Long Sơn | 25MS13300 |
| Đoàn Đàm Quân | 25MS13305 |
| Lê Thanh Phương Nam | 25MS23308 |
| Hoàng Quốc Anh | - |

Môn học: **DDM501.22 — AI trong sản xuất: DevOps, DataOps, MLOps**

---

## 1. Mục tiêu

Lab 4 triển khai một hệ thống credit risk scoring có khả năng giám sát trong production.

Hệ thống theo dõi:

- Sức khỏe API.
- HTTP request count và latency.
- Input drift bằng Population Stability Index (PSI).
- Phân phối prediction score.
- Decision mix: APPROVE, REVIEW, DECLINE.
- Selection rate theo nhóm.
- Fairness gap.
- Prediction errors.
- SHAP explanation latency.
- Alert bằng Prometheus.
- Dashboard bằng Grafana.
- Đóng gói toàn bộ hệ thống bằng Docker Compose.

Accuracy không phải tín hiệu production chính trong bài này vì nhãn default xuất hiện trễ. Với hồ sơ bị từ chối, repayment outcome không tồn tại. Vì vậy hệ thống sử dụng các tín hiệu có thể quan sát ngay khi service đang chạy.

---

## 2. Các tín hiệu giám sát

| Tín hiệu | Câu hỏi | Metric |
|---|---|---|
| Input drift | Applicant có còn giống dữ liệu training không? | `ml_feature_drift_psi` |
| Output drift | Phân phối điểm dự đoán có thay đổi không? | `ml_prediction_score` |
| Decision mix | Tỷ lệ REVIEW/DECLINE có thay đổi không? | `ml_decisions_total` |
| Fairness | Các nhóm có selection rate khác nhau không? | `ml_fairness_gap` |
| Service health | API có hoạt động và đáp ứng đủ nhanh không? | `http_requests_total`, `http_request_duration_seconds` |

PSI được đọc theo quy ước:

| PSI | Ý nghĩa |
|---:|---|
| `< 0.10` | Stable |
| `0.10 - < 0.25` | Moderate drift |
| `>= 0.25` | Significant drift |

---

## 3. Kiến trúc Docker

```text
Load Test
    |
    v
FastAPI Credit Risk API
    |
    +--> Prometheus
    |       |
    |       +--> Alert rules
    |
    +--> Grafana dashboards
    |
    +--> Node Exporter
```

Docker Compose gồm bốn service:

| Service | Port | Chức năng |
|---|---:|---|
| `api` | 8000 | FastAPI model serving |
| `prometheus` | 9090 | Thu thập metric và đánh giá alert |
| `grafana` | 3000 | Hiển thị dashboard |
| `node-exporter` | 9100 | Host metrics |

Prometheus scrape API qua Docker service name:

```text
api:8000
```

Không dùng IP cố định trong cấu hình.

---

## 4. Cấu trúc project

```text
app/
  main.py
  metrics.py
  middleware.py
  monitoring.py
  explain.py
  model.py
  schemas.py

pipeline/
  data_ingestion.py
  preprocessing.py
  training.py
  validation.py

scripts/
  make_dataset.py
  train_model.py
  make_reference.py
  load_test.py

monitoring/
  prometheus/
    alerts/
    tests/
    prometheus.yml
  grafana/
    dashboards/
    provisioning/

tests/
Dockerfile
docker-compose.yml
requirements.txt
```

---

## 5. Yêu cầu môi trường

- Docker Desktop
- Docker Compose v2+
- Python 3.11+
- Git

Kiểm tra:

```powershell
docker version
docker compose version
python --version
git --version
```

---

## 6. Cài đặt local

Tạo virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu đã có `.venv`, có thể dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 7. Tạo dataset, model và reference

Dataset được tạo offline để project có thể chạy mà không phụ thuộc vào việc tải UCI dataset từ mạng.

```powershell
python -m scripts.make_dataset
python -m scripts.train_model
python -m scripts.make_reference
```

Hoặc không activate virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.make_dataset
.\.venv\Scripts\python.exe -m scripts.train_model
.\.venv\Scripts\python.exe -m scripts.make_reference
```

Kết quả huấn luyện hiện tại:

```text
Rows: 30,000
Training rows: 24,000
Test rows: 6,000
ROC AUC: approximately 0.747
PR AUC: approximately 0.544
```

Reference drift được tạo từ training split và lưu tại:

```text
models/reference.json
```

---

## 8. Chạy test

Chạy toàn bộ test:

```powershell
python -m pytest -q
```

Kết quả xác nhận của nhóm:

```text
79 passed
Coverage: 90.50%
```

Các nhóm test:

```powershell
python -m pytest tests/test_monitoring.py --no-cov -q
python -m pytest tests/test_metrics.py --no-cov -q
python -m pytest tests/test_api_monitoring.py --no-cov -q
python -m pytest tests/test_config_files.py --no-cov -q
```

Coverage tối thiểu được cấu hình trong `pytest.ini` là 85%.

---

## 9. Kiểm tra Prometheus alert rules

Kiểm tra cú pháp rule:

```powershell
docker run --rm `
    --entrypoint promtool `
    -v "${PWD}:/work" `
    -w /work `
    prom/prometheus:v2.51.2 `
    check rules `
    monitoring/prometheus/alerts/api_alerts.yml `
    monitoring/prometheus/alerts/ml_alerts.yml
```

Chạy unit test alert:

```powershell
docker run --rm `
    --entrypoint promtool `
    -v "${PWD}:/work" `
    -w /work/monitoring/prometheus/tests `
    prom/prometheus:v2.51.2 `
    test rules alert_tests.yml
```

Kết quả xác nhận:

```text
API rules: 5 rules found
ML rules: 7 rules found
Unit Testing: SUCCESS
```

Các alert ML:

- `ModerateFeatureDrift`
- `SignificantFeatureDrift`
- `DriftWindowTooSmall`
- `DecisionMixShift`
- `FairnessGapWidened`
- `PredictionErrorsRising`
- `ExplanationLatencyHigh`

---

## 10. Chạy bằng Docker Compose

Kiểm tra Compose:

```powershell
docker compose config --quiet
```

Build và khởi động:

```powershell
docker compose up -d --build
```

Kiểm tra container:

```powershell
docker compose ps
```

Dừng hệ thống:

```powershell
docker compose down
```

Dừng và xóa volumes:

```powershell
docker compose down -v
```

---

## 11. Các địa chỉ truy cập

| Thành phần | Địa chỉ |
|---|---|
| Swagger API | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |
| Monitoring JSON | http://localhost:8000/monitoring |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana mặc định:

```text
Username: admin
Password: admin
```

Kiểm tra API:

```powershell
Invoke-RestMethod "http://localhost:8000/health" |
    ConvertTo-Json

Invoke-RestMethod "http://localhost:8000/monitoring" |
    ConvertTo-Json -Depth 8
```

Kiểm tra Prometheus scrape API:

```powershell
Invoke-RestMethod `
    "http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22credit-risk-api%22%7D" |
    ConvertTo-Json -Depth 8
```

Kết quả cần có giá trị:

```text
up = 1
```

---

## 12. Chạy load test

Luôn chạy script dưới dạng module để Python nhận diện package `pipeline`:

```powershell
python -m scripts.load_test
```

### Normal traffic

```powershell
python -m scripts.load_test `
    --profile normal `
    --requests 400 `
    --delay 0
```

Kỳ vọng:

```text
drift_status: stable
```

### Mild drift

```powershell
docker compose restart api
Start-Sleep -Seconds 8

python -m scripts.load_test `
    --profile drifted `
    --strength 0.05 `
    --requests 400 `
    --delay 0
```

Kỳ vọng: drift score thuộc vùng moderate.

### Significant drift

```powershell
docker compose restart api
Start-Sleep -Seconds 8

python -m scripts.load_test `
    --profile drifted `
    --strength 1.0 `
    --requests 400 `
    --delay 0
```

Kỳ vọng:

```text
drift_status: significant
drift_score > 0.25
```

### Fairness shift

```powershell
docker compose restart api
Start-Sleep -Seconds 8

python -m scripts.load_test `
    --profile unfair `
    --requests 400 `
    --delay 0
```

Kỳ vọng:

```text
selection_rate có hai nhóm
fairness_gap > 0.10
```

Xem monitoring state sau mỗi profile:

```powershell
Invoke-RestMethod "http://localhost:8000/monitoring" |
    ConvertTo-Json -Depth 8
```

---

## 13. Các metric chính

| Metric | Ý nghĩa |
|---|---|
| `http_requests_total` | Tổng số HTTP request |
| `http_request_duration_seconds` | HTTP latency |
| `ml_predictions_total` | Tổng số prediction |
| `ml_prediction_score` | Phân phối score |
| `ml_feature_drift_psi` | PSI theo feature |
| `ml_drift_score` | PSI lớn nhất |
| `ml_drift_window_size` | Số request trong monitoring window |
| `ml_decisions_total` | Số lượng decision theo loại |
| `ml_selection_rate` | Selection rate theo nhóm |
| `ml_fairness_gap` | Khoảng cách selection rate |
| `ml_prediction_errors_total` | Lỗi prediction |
| `ml_explain_duration_seconds` | SHAP explanation latency |
| `ml_model_loaded` | Trạng thái model |

---

## 14. Grafana dashboards

Dashboard được provision từ các file trong Git:

```text
monitoring/grafana/dashboards/service-health.json
monitoring/grafana/dashboards/model-behaviour.json
```

Model Behaviour dashboard tập trung vào:

- Maximum PSI.
- PSI theo từng feature.
- Monitoring window size.
- Prediction score distribution.
- Decision mix.
- Selection rate theo nhóm.
- Fairness gap.
- SHAP explanation latency.
- Prediction errors.

Dashboard được lưu trong source code để các môi trường Docker sử dụng cùng một cấu hình.

---

## 15. Mười ba task của Lab 4

| # | File | Nội dung |
|---:|---|---|
| 1 | `app/monitoring.py` | PSI formula |
| 2 | `app/monitoring.py` | Feature drift |
| 3 | `app/monitoring.py` | Fairness |
| 4 | `app/monitoring.py` | Publish monitoring metrics |
| 5 | `app/main.py` | Observe predictions |
| 6 | `app/main.py` | `GET /metrics` |
| 7 | `app/main.py` | `GET /monitoring` |
| 8 | `app/main.py` | `POST /explain` |
| 9 | `app/middleware.py` | HTTP metrics middleware |
| 10 | `app/explain.py` | SHAP explanation |
| 11 | `scripts/make_reference.py` | Frozen reference |
| 12 | `ml_alerts.yml` | Seven ML alert rules |
| 13 | `model-behaviour.json` | Model monitoring dashboard |

---

## 16. Nguyên tắc MLOps được áp dụng

- Reference được tạo từ training split.
- Reference được đóng băng cùng model.
- Quantile bins được dùng để tính PSI.
- Giá trị ngoài training range được giữ trong outer bins.
- Monitoring window có giới hạn kích thước.
- Dữ liệu chưa đủ được báo là `sufficient_data = false`.
- Derived features được tạo trước khi ghi vào monitoring window.
- Counter, Gauge và Histogram được dùng theo đúng ý nghĩa.
- HTTP metric dùng route template để tránh cardinality explosion.
- Container API chạy bằng non-root user.
- Prometheus và Grafana được provision từ file version-controlled.
- Alert có threshold, `for`, severity và hướng dẫn xử lý.

---

## 17. Kết quả xác nhận

Các kiểm tra đã hoàn tất:

```text
Python tests: 79 passed
Coverage: 90.50%
Prometheus rule syntax: SUCCESS
Prometheus alert tests: SUCCESS
Docker Compose config: valid
API container: healthy
Prometheus scrape: up = 1
Grafana: database ok
```

---

## 18. Tài liệu tham khảo

- `DDM501_Lab4_Monitoring.pdf`
- FSB Lesson 01 — Introduction to ML Production
- FSB Lesson 02 — Requirements Engineering and Setting Goals
- FSB Lesson 04 — Architecture and System Design
- FSB Lesson 06 — Data Quality and Model Quality
- FSB Lesson 08 — Deployment Strategies
- Chip Huyen — Designing Machine Learning Systems
- Geoff Hulten — Building Intelligent Systems

## Kết quả kiểm chứng và phân tích

- [Phân tích monitoring - Nhóm 7](docs/Lab4_Monitoring_Analysis_Group7.pdf)
- [Dashboard normal](docs/evidence/01_normal_dashboard.png)
- [Dashboard drifted](docs/evidence/02_drifted_dashboard.png)
- [Dashboard unfair](docs/evidence/03_unfair_dashboard.png)

Traffic logs và monitoring JSON nằm trong `docs/evidence/`.
Mỗi profile dùng 400 requests, seed 501, reset API giữa các lượt.
Phân tích phân biệt điều kiện vượt ngưỡng với alert thực sự firing.
