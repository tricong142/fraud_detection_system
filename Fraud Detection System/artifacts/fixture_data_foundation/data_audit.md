# Data audit — SYNTHETIC TEST FIXTURE; NOT PAYSIM

Số liệu dưới đây được tính từ file có SHA-256 trong `dataset_manifest.json`.
Nguồn khai báo: https://www.kaggle.com/datasets/ealaxi/paysim1 .
Checksum xác định file đã xử lý; tự nó không chứng minh file đến từ Kaggle.

## 1. Kích thước, thời gian, tài khoản và cạnh

| metric | value |
| --- | --- |
| rows | 41.0 |
| columns | 11.0 |
| duplicates_excluding_first | 1.0 |
| fraud | 20.0 |
| nonfraud | 21.0 |
| fraud_rate | 0.4878048780487805 |
| step_min | 1.0 |
| step_max | 20.0 |
| observed_steps | 20.0 |
| missing_hours_inside_range | 0.0 |
| unique_origin_accounts | 21.0 |
| unique_destination_accounts | 2.0 |
| unique_accounts_any_role | 22.0 |
| accounts_in_both_roles | 1.0 |
| origins_with_multiple_transactions | 1.0 |
| destinations_with_multiple_transactions | 2.0 |
| max_outgoing_transactions | 21.0 |
| max_incoming_transactions | 21.0 |
| unique_directed_edges | 21.0 |
| repeated_directed_edges | 20.0 |
| self_loop_transactions | 0.0 |
| directed_density_excluding_self_loops | 0.045454545454545456 |
| mean_out_degree_unique | 0.9545454545454546 |

## 2. Schema và missing values

| column | dtype | missing |
| --- | --- | --- |
| step | int64 | 0 |
| type | string | 0 |
| amount | float64 | 0 |
| nameOrig | string | 0 |
| oldbalanceOrg | float64 | 0 |
| newbalanceOrig | float64 | 0 |
| nameDest | string | 0 |
| oldbalanceDest | float64 | 0 |
| newbalanceDest | float64 | 0 |
| isFraud | int8 | 0 |
| isFlaggedFraud | int8 | 0 |

Dtype là contract dùng để đọc dữ liệu, không phải dtype suy đoán của CSV.
Duplicates được đếm theo toàn bộ 11 cột đã parse, xuyên chunk; không tự xóa.
Thiếu trường bắt buộc, label không nhị phân, amount âm/vô hạn hoặc step sai thứ tự sẽ làm pipeline dừng.
Balance thiếu được ghi nhận; balance không thuộc feature huấn luyện MVP.

## 3. Phân bố nhãn theo loại giao dịch

| type | transactions | fraud | nonfraud | fraud_rate |
| --- | --- | --- | --- | --- |
| PAYMENT | 21 | 0 | 21 | 0.0 |
| TRANSFER | 20 | 20 | 0 | 1.0 |

Phân bố theo từng giờ: `by_step.csv`; theo giờ và loại: `by_step_type.csv`.
Các file này gồm số giao dịch, fraud, nonfraud và fraud rate.
`by_step.csv` có tỷ trọng giao dịch/fraud theo giờ để nhận diện tập trung thời gian.

## 4. Khả năng sử dụng lịch sử tài khoản/graph

`account_activity.parquet`: mỗi account có outgoing_count, incoming_count và thời điểm
xuất hiện lần đầu ở từng vai trò. `account_count_distribution.csv` cho phân bố số giao dịch.
Đây là thống kê toàn dataset phục vụ audit, KHÔNG được join trực tiếp vào training features.
Tài khoản mới ở step t có first_seen_step = t, xét cả nguồn và đích của toàn bộ step;
mọi giao dịch cùng step đều coi tài khoản này là chưa có lịch sử.
`new_origin_accounts`/`new_destination_accounts` chỉ lần đầu ở vai trò tương ứng;
`new_accounts` chỉ lần đầu ở bất kỳ vai trò nào, không cộng hai cột vai trò.
Một self-loop được tính một lần outgoing và một lần incoming.
Density là đồ thị có hướng đơn trên toàn bộ node, bỏ self-loop; repeated edges giữ nguyên trong raw.
Mật độ tổng thể không chứng minh một graph feature có ích: cần ablation E2/E3 trên cùng split.

## 5. Temporal split đã khóa

| split | step_min | step_max | steps | rows | fraud | fraud_rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 14 | 14 | 29 | 14 | 0.4827586206896552 |
| validation | 15 | 17 | 3 | 6 | 3 | 0.5 |
| test | 18 | 20 | 3 | 6 | 3 | 0.5 |

Tỷ lệ áp dụng cho số step quan sát được, dùng floor ở mốc train và train+validation;
không áp dụng cho số dòng. Các giờ không có giao dịch vẫn được báo cáo là gap.
Mọi split phải có cả fraud và nonfraud; nếu không, pipeline dừng và không công bố output.
Không tự dò ranh giới bằng test labels. Thay ranh giới cần quyết định có ghi nhận trước đánh giá.
Mọi E1/E2/E3 dùng chung `split_definition.json` và checksum file nguồn.
Phân bố khác nhau giữa train/validation/test cần được xem xét trước khi train;
không tự cân bằng hoặc chọn lại test để cải thiện metric.

## 6. Point-in-time contract

History(t) = {x | x.step < t}. Score toàn bộ step trên state đã đóng băng,
sau đó mới cập nhật state. Không suy ra thứ tự trong giờ từ row_id.
Không dùng label trong history features. Lịch sử giao dịch train được phép warm-up validation,
validation warm-up test; không truyền labels sang feature state.
Fixtures chỉ kiểm tra logic; chưa đủ để kết luận về PaySim hoặc hiệu quả mô hình.
