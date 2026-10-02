# DDM501.22 — Individual Assignment 1

## Thông tin học viên

| Nội dung | Thông tin |
|---|---|
| Họ tên | Lê Thanh Phương Nam |
| Mã học viên | 25MS23308 |
| Môn học | DDM501.22 — AI trong sản xuất: DevOps, DataOps, MLOps |
| Bài nộp | Individual Assignment 1 |
| Nhánh Git | `assignment1` |

## 1. Mục đích của nhánh

Nhánh này tập hợp source code tham chiếu phục vụ bài cá nhân Assignment 1
về thiết kế hệ thống dự đoán rủi ro tín dụng.

Báo cáo cá nhân trình bày bài toán, yêu cầu, mục tiêu và kiến trúc hệ thống.
Source code cung cấp môi trường minh họa cho các thành phần như model
inference, API, monitoring và Docker deployment.

Code nền được kế thừa từ nhánh `main`, nơi lưu bài thực hành của nhóm.
Việc tạo nhánh riêng giúp phân biệt nội dung nộp cá nhân với bài nhóm;
không coi toàn bộ code kế thừa là phần phát triển mới của cá nhân.

## 2. Phân biệt các bài nộp

| Nhánh | Mục đích |
|---|---|
| `main` | Bài thực hành monitoring của nhóm |
| `assignment1` | Source tham chiếu và tài liệu cho Individual Assignment 1 |
| `assignment2` | Experiment tracking, MLflow và Airflow cho Individual Assignment 2 |

- [Bài nhóm — main](https://github.com/NamBobby/ddm501-lab4-starter/tree/main)
- [Bài cá nhân — Assignment 1](https://github.com/NamBobby/ddm501-lab4-starter/tree/assignment1)
- [Bài cá nhân — Assignment 2](https://github.com/NamBobby/ddm501-lab4-starter/tree/assignment2)

## 3. Hệ thống tham chiếu

Hệ thống credit risk scoring gồm:

- Pipeline preprocessing và mô hình dự đoán credit default.
- FastAPI cung cấp prediction, explanation và health endpoints.
- Theo dõi input drift, decision mix và fairness gap.
- Prometheus thu thập metrics và đánh giá alert rules.
- Grafana hiển thị dashboards.
- Docker Compose khởi động các services.

Các thành phần triển khai này hỗ trợ minh họa thiết kế.
Phạm vi và các đề xuất mở rộng của Assignment 1 được trình bày trong
báo cáo cá nhân, không mặc nhiên coi mọi đề xuất là đã triển khai.

## 4. Clone đúng bài cá nhân

```powershell
git clone --branch assignment1 --single-branch https://github.com/NamBobby/ddm501-lab4-starter.git ddm501-assignment1
Set-Location ".\ddm501-assignment1"
```

## 5. Chạy bằng Docker

Yêu cầu: Docker Desktop đang chạy Linux containers và Docker Compose.

```powershell
docker compose up -d --build
docker compose ps
```

| Thành phần | Địa chỉ |
|---|---|
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |
| Monitoring JSON | http://localhost:8000/monitoring |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana dùng tài khoản demo `admin` / `admin`.

Kiểm tra API:

```powershell
Invoke-RestMethod "http://localhost:8000/health"
Invoke-RestMethod "http://localhost:8000/monitoring"
```

Xem logs và dừng services:

```powershell
docker compose logs --tail 100 api
docker compose down
```

## 6. Chuẩn bị model nếu cần

Yêu cầu Python 3.11. Chỉ thực hiện nếu cần tạo lại dữ liệu hoặc model.

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m scripts.make_dataset
& .\.venv\Scripts\python.exe -m scripts.train_model
& .\.venv\Scripts\python.exe -m scripts.make_reference
```

Sau khi tạo lại model và reference, khởi động lại API:

```powershell
docker compose restart api
```

## 7. Cấu trúc source

| Thư mục / file | Vai trò |
|---|---|
| `app/` | API, metrics và monitoring |
| `pipeline/` | Các thành phần xử lý dữ liệu và mô hình |
| `scripts/` | Dataset, training, reference và traffic generation |
| `models/` | Model và drift reference |
| `monitoring/` | Prometheus, alert rules và Grafana dashboards |
| `tests/` | Kiểm thử hệ thống tham chiếu |
| `Dockerfile` | Image của API |
| `docker-compose.yml` | Cấu hình các services |

## 8. Báo cáo cá nhân

Tên file nộp:

`DDM501_Assignment1_25MS23308_LeThanhPhuongNam.pdf`

Báo cáo PDF được nộp riêng theo yêu cầu môn học.
Nếu bổ sung PDF vào repository, đặt trong `docs/report/`.

## 9. Nguồn kế thừa

Source nền được kế thừa từ bài thực hành của nhóm trong cùng repository.
Nhánh `assignment1` tổ chức riêng phần tham chiếu cho báo cáo cá nhân;
nhánh `assignment2` lưu phần phát triển tiếp theo về ML pipeline và MLOps.
