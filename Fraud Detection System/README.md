# Fraud Detection System — Giai đoạn 1: Data Foundation

Phạm vi hiện tại: lấy CSV gốc PaySim, kiểm tra dữ liệu, khóa chính sách chống leakage,
chia train/validation/test theo thời gian. Chưa huấn luyện model hoặc tạo behavioral/graph features.

## 1. Kiến trúc modular monolith

Một Python package, một tiến trình offline; `data_foundation` là module nghiệp vụ đầu tiên.
Các giai đoạn sau chỉ thêm module khi triển khai, chưa tạo service rỗng.

```text
scripts/
  00_download_paysim.py              # CLI tải dữ liệu, không chứa credential
  01_data_audit.py                   # CLI nhận config và gọi application service
src/fraud_detection/data_foundation/
  contract.py                       # schema, split policy, history/feature contract
  storage.py                        # repository SQLite tạm + export Parquet
  audit.py                          # thống kê, báo cáo EDA
  pipeline.py                       # application service điều phối workflow
  feature_policy.md                 # chính sách có version, được copy vào output
tests/test_data_foundation.py
```

Áp dụng Application Service và Repository ở mức module/hàm, không thêm abstract class
hoặc framework DI. CLI không chứa nghiệp vụ. `SplitPolicy` là cấu hình bất biến có validation.
Đây là lựa chọn triển khai nhỏ cho yêu cầu hiện tại, không phải khẳng định có một pattern
duy nhất đúng. Offline foundation tách trách nhiệm với online inference ở giai đoạn sau.

Luồng: CSV → validate theo chunk → SQLite trên đĩa → audit → khóa split theo step
→ xuất Parquet và report → checksum/manifest → công bố thư mục output.
SQLite chỉ là vùng staging cho audit, không phải database giao dịch của app.

## 2. Cài môi trường

PowerShell, chạy từ thư mục gốc dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Python >= 3.10. Pip chọn bản Kaggle tương thích với Python; kiểm tra `kaggle --help`
nếu bản được cài có cơ chế xác thực khác. Dependency hiện là khoảng tương thích,
chưa phải lockfile cho triển khai production. Manifest ghi version thực tế dùng khi audit.

## 3. Tải CSV gốc

Nguồn: [PaySim — ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1).
Theo [tài liệu Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md),
credentials có thể cấu hình ngoài repository, ví dụ `$HOME/.kaggle/kaggle.json`
(legacy) hoặc token theo hướng dẫn của phiên bản CLI. Không gửi token qua chat hoặc
đưa vào source/.env được commit. Không tự tạo credentials giả.

```powershell
.\.venv\Scripts\python.exe scripts/00_download_paysim.py
```

Script gọi đúng Kaggle CLI tương đương:

```powershell
.\.venv\Scripts\kaggle.exe datasets download -d ealaxi/paysim1 -p data/raw --unzip
```

Wrapper không ghi đè thư mục raw có dữ liệu, xác minh chỉ có một CSV, tính checksum
và ghi `download_receipt.json` sau khi download thành công. Nếu tải thủ công thì
audit vẫn hoạt động, manifest ghi nguồn do người dùng khai báo và chưa xác thực.
Không hard-code filename hoặc số dòng từ bài hướng dẫn.

Trong môi trường đã kiểm tra, Kaggle 1.7.4.5 báo `KeyError: 'username'` khi tải
không có credentials. Cần cấu hình tài khoản thật trước khi chạy; không dùng
username/key giả để bỏ qua lỗi. Chưa có CSV thì chưa có audit PaySim thực tế.

## 4. Chạy audit và chia dữ liệu

```powershell
.\.venv\Scripts\python.exe scripts/01_data_audit.py --chunksize 100000
```

Nếu nhiều CSV, truyền đường dẫn cụ thể:

```powershell
.\.venv\Scripts\python.exe scripts/01_data_audit.py --csv data/raw/ten_file_thuc_te.csv --output-dir artifacts/paysim_v1
```

Pipeline yêu cầu raw đã sắp xếp step tăng không giảm. Không tự shuffle/sort hoặc
xóa duplicate. Nếu input không đúng contract, lệnh trả exit code khác 0.
`--fixture` chỉ dùng với dữ liệu test tự tạo, luôn gắn nhãn rõ trong báo cáo.

Output mặc định `artifacts/data_foundation/`:

| File | Ý nghĩa |
| --- | --- |
| data_audit.md | Báo cáo kích thước, schema, missing, duplicate, tài khoản, graph và split |
| audit_summary.json | Các số đo đọc được bằng chương trình |
| by_step.csv / by_type.csv / by_step_type.csv | Phân bố giao dịch và nhãn theo thời gian/loại |
| account_activity.parquet | Số giao dịch vào/ra và first-seen của mỗi account, chỉ dùng EDA |
| account_count_distribution.csv | Phân bố số giao dịch trên mỗi tài khoản |
| feature_policy.md | Bản chính sách đi kèm đúng lần audit |
| split_definition.json | Step cụ thể, lý do chia và checksum chung E1/E2/E3 |
| splits/train.parquet | Dữ liệu train, các cột raw + row_id |
| splits/validation.parquet | Dữ liệu tuning/threshold |
| splits/test.parquet | Test cuối, dùng chung các thí nghiệm |
| dataset_manifest.json | SHA-256 nguồn và outputs, version runtime, trạng thái provenance |

`row_id` là số thứ tự dòng gốc từ 0, chỉ dùng đối chiếu/tái lập, không phải feature.
Manifest không chứa checksum chính nó. Checksum xác định bytes, không chứng minh nguồn Kaggle.

## 5. Các quyết định cần hiểu

- Chia theo **số step quan sát được**: 70%/15%/15%, dùng floor ở hai mốc tích lũy.
  Tỷ lệ số dòng/fraud có thể khác xa tỷ lệ này. Báo cáo từng tập để nhận diện điều đó.
- Từng tập cần cả fraud/nonfraud. Nếu không đủ, dừng; không âm thầm dịch test boundary.
  Muốn thay tỷ lệ, dùng `--train-ratio`, `--validation-ratio`, `--split-reason` và
  output mới. Quyết định phải có trước khi đánh giá model, không chọn theo test metric.
- Mặc định E1 chỉ được chọn `amount`, `type`; label, flagged fraud, balance và raw ID
  bị loại bằng allowlist. Thêm step cần ablation riêng.
- History của t chỉ gồm step < t. Cùng step score trước, update state sau.
  Thống kê account toàn kỳ của EDA tuyệt đối không join làm feature.
- RAM được hạn chế bằng chunksize và SQLite trên đĩa; không hứa RAM hằng số tuyệt đối.
  Cần chừa dung lượng cho raw, SQLite/temp sort và ba Parquet; peak phụ thuộc dữ liệu.
  Disk aggregation chậm hơn in-memory. Đừng bắt đầu graph feature trước khi xem mức lặp account.
- Thư mục output đã tồn tại bị từ chối để bảo vệ split. Rerun dùng tên mới.
- PaySim là dữ liệu mô phỏng theo [tác giả](https://github.com/EdgarLopezPhD/PaySim);
  chưa thể suy ra hiệu quả trên giao dịch tài chính thực tế.

Đọc chi tiết tại [feature_policy.md](src/fraud_detection/data_foundation/feature_policy.md).

## 6. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests dùng fixture tổng hợp và đối chiếu pandas cho audit/splits; kiểm tra same-step
history, duplicate xuyên chunk, account role overlap, label overflow, balance missing,
single-class split, schema, thứ tự thời gian, checksum và output bất biến.
Test pass trên fixture không đồng nghĩa đã audit PaySim thật hay đạt chất lượng model.

## 7. Trạng thái đã xác minh tại workspace này

- 11 regression tests pass trên Windows, Python 3.10.11, pandas 2.2.2,
  pyarrow 23.0.1. Đã kiểm tra compile Python và CLI `--help`.
- CLI đã chạy hoàn chỉnh với fixture 41 dòng/20 step; output ở
  `artifacts/fixture_data_foundation/`, gắn nhãn rõ **SYNTHETIC TEST FIXTURE; NOT PAYSIM**.
- Chưa tải được PaySim thật: máy chưa có credentials; Kaggle CLI 1.7.4.5
  dừng với `KeyError: 'username'`. Chưa có số đo EDA, thời gian chạy hoặc RAM
  thực tế của full dataset; không dùng số liệu fixture để suy luận chúng.
- Môi trường kiểm tra tận dụng package có sẵn bằng `--system-site-packages`.
  `pip check` còn báo xung đột protobuf 3.20.2 với một số package Google được kế
  thừa từ Python hệ thống. Các package đó không thuộc pipeline audit, nhưng môi
  trường hiện tại chưa sạch dependency. Hướng dẫn cài ở mục 2 tạo venv độc lập;
  cần chạy lại tests và `pip check` trong môi trường độc lập trước deployment.
- Khi có credentials, chạy script download rồi audit theo mục 3–4.
  Giai đoạn 1 chỉ được nghiệm thu trên dữ liệu thật khi đã xem đủ báo cáo,
  phân bố nhãn theo thời gian và chốt split chung cho E1/E2/E3.

