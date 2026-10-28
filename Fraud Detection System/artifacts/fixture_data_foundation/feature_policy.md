# Feature policy — MVP, contract 1.0.0

## Quyết định trước khi feature engineering

Nguồn khai báo: [PaySim trên Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
Theo cảnh báo tác giả được cung cấp trong yêu cầu dự án, giao dịch được phát hiện
gian lận có thể bị hủy; bốn trường balance không phù hợp cho fraud detection.
MVP áp dụng quy tắc loại trừ dưới đây. Không coi mọi trường có trong CSV là đầu vào
có sẵn tại thời điểm dự đoán. Raw và split vẫn giữ nguyên các cột để audit/reproduce.

| Trường | Vai trò | Được làm feature MVP? |
| --- | --- | --- |
| amount | Giá trị giao dịch đang yêu cầu | Có |
| type | Loại giao dịch đang yêu cầu | Có; encoder chỉ fit trên train |
| isFraud | Target của supervised learning | Không |
| isFlaggedFraud | Cờ của rule mô phỏng | Không |
| oldbalanceOrg, newbalanceOrig | Số dư nguồn | Không; kể cả feature suy ra từ balance |
| oldbalanceDest, newbalanceDest | Số dư đích | Không; kể cả feature suy ra từ balance |
| nameOrig, nameDest | Khóa tra cứu lịch sử | Không đưa categorical ID trực tiếp vào model |
| step | Thời gian/cửa sổ lịch sử | Không ở baseline mặc định; chỉ thêm trong ablation có ghi nhận |
| row_id | Vị trí dòng gốc, 0-based | Không; không có ý nghĩa thứ tự trong cùng giờ |

`baseline_features()` dùng allowlist amount/type; tùy chọn include_step là thí nghiệm
riêng. Helper này chưa encode type và chưa huấn luyện model. Hợp đồng input online
ở giai đoạn sau phải giữ cùng ý nghĩa amount/type; cần xác minh với API thực tế.

## Hợp đồng thời gian

Quy ước dự án: một step là một giờ mô phỏng, không có timestamp nhỏ hơn giờ.

**History(t) = {transaction | transaction.step < t}.**

1. Nhận batch mọi giao dịch của step t.
2. Đọc cùng state lịch sử đã kết thúc ở các step trước t.
3. Tạo feature và score toàn batch; không cập nhật state sau từng dòng.
4. Sau khi score xong batch, cập nhật state bằng giao dịch của step t.

`history_before()` minh họa và kiểm thử điều kiện `<`; behavioral/graph engine
chưa được triển khai ở giai đoạn này. Offline train/val/test và online replay
phải thực thi cùng quy tắc batch. Nếu sau này có timestamp chi tiết hơn thì đổi
contract có version, không âm thầm dùng row order làm timestamp.

- Tài khoản lần đầu xuất hiện ở t là chưa có lịch sử cho mọi dòng trong t.
- Dùng cùng namespace ID giữa nguồn/đích; ID giống nhau là cùng node.
- Có thể warm-up validation bằng giao dịch train, test bằng giao dịch train+validation.
- Trong validation/test được cập nhật state bằng các giao dịch **đã qua**, theo cùng
  cơ chế online; không học lại model, encoder, scaler hoặc threshold bằng test.
- Không đưa isFraud/isFlaggedFraud vào behavioral/graph state. Chưa có mô hình
  label delay nên không dùng lịch sử nhãn, kể cả nhãn của các step trước.
- Thống kê toàn kỳ trong account_activity.parquet chỉ phục vụ EDA. Join các số đếm
  này vào training sẽ nhìn thấy tương lai và vi phạm contract.

## Chia dữ liệu và đánh giá

- Sắp xếp các **step quan sát được**; floor(N * 0.70) step đầu cho train,
  đến floor(N * 0.85) cho validation, phần còn lại cho test. Không shuffle dòng.
- Không giả định số dòng, số step hay tên file từ mô tả: đo từ dữ liệu thực tế.
- Các giờ không có dữ liệu được báo cáo là gap; tỷ lệ dùng số step quan sát,
  không phải độ dài lịch bao gồm giờ trống. Quy tắc này có trong manifest split.
- Mỗi tập cần có cả hai nhãn để kiểm tra/đánh giá phân loại; đây là kiểm tra tối thiểu,
  không chứng minh số fraud đã đủ cho đánh giá ổn định.
- Nếu fail, xem phân bố thời gian, ghi lý do thay tỷ lệ bằng `--split-reason`, dùng
  output directory mới. Không tự tìm split tối ưu theo test metric.
- E1/E2/E3 dùng nguyên split_definition.json, raw checksum và test set.
- Fit preprocessing/model trên train, tune hyperparameters/threshold trên validation,
  chỉ đánh giá test khi thiết kế đã chốt. Báo cáo PR-AUC, precision/recall và confusion
  matrix theo threshold ở giai đoạn model; không dùng accuracy một mình.
- Audit có đọc nhãn test để kiểm tra chất lượng/phân bố theo yêu cầu; đây không phải
  test blind tuyệt đối. Không tận dụng kết quả này để tối ưu model theo test.

## Giới hạn dữ liệu

PaySim là dữ liệu mô phỏng theo [repository của tác giả](https://github.com/EdgarLopezPhD/PaySim).
Kết quả benchmark không chứng minh model tổng quát tốt trên giao dịch thực tế.
Không tự loại duplicate: giao dịch giống hệt nhau chưa chắc là dữ liệu lỗi.
Float64 giữ quy ước phân tích CSV này; API giao dịch tiền thật cần đơn vị tiền và
quy tắc làm tròn rõ ràng, không lấy float CSV làm hợp đồng thanh toán production.
SHA-256 giúp tái lập file; không tự chứng minh nguồn gốc hay version Kaggle.

