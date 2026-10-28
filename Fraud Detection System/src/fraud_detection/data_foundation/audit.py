"""Descriptive audit only: labels here are never passed to feature generation."""

from pathlib import Path
import sqlite3

import pandas as pd

from .contract import COLUMNS


def collect_audit(connection: sqlite3.Connection, output: Path, basic: dict) -> tuple[dict, pd.DataFrame]:
    columns = ",".join(f'"{name}"' for name in COLUMNS)
    duplicate_rows = connection.execute(f"""
        SELECT COALESCE(SUM(n - 1), 0) FROM (
            SELECT COUNT(*) AS n FROM transactions GROUP BY {columns} HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    by_step = pd.read_sql_query("""
        SELECT step, COUNT(*) AS transactions, SUM(isFraud) AS fraud
        FROM transactions GROUP BY step ORDER BY step
    """, connection)
    by_type = pd.read_sql_query("""
        SELECT type, COUNT(*) AS transactions, SUM(isFraud) AS fraud
        FROM transactions GROUP BY type ORDER BY type
    """, connection)
    by_step_type = pd.read_sql_query("""
        SELECT step, type, COUNT(*) AS transactions, SUM(isFraud) AS fraud
        FROM transactions GROUP BY step, type ORDER BY step, type
    """, connection)
    for table in (by_step, by_type, by_step_type):
        table["nonfraud"] = table["transactions"] - table["fraud"]
        table["fraud_rate"] = table["fraud"] / table["transactions"]
    for field, name in (("first_seen_step", "new_accounts"),
                        ("first_origin_step", "new_origin_accounts"),
                        ("first_destination_step", "new_destination_accounts")):
        new = pd.read_sql_query(
            f"SELECT {field} AS step, COUNT(*) AS {name} FROM accounts "
            f"WHERE {field} IS NOT NULL GROUP BY {field}", connection
        )
        by_step = by_step.merge(new, on="step", how="left")
        by_step[name] = by_step[name].fillna(0).astype("int64")
    by_step["prior_seen_accounts"] = by_step["new_accounts"].cumsum() - by_step["new_accounts"]
    by_step["transaction_share"] = by_step["transactions"] / basic["rows"]
    total_fraud = int(by_step["fraud"].sum())
    by_step["fraud_share"] = by_step["fraud"] / total_fraud if total_fraud else 0.0
    summary = connection.execute("""
        SELECT COUNT(*), SUM(outgoing_count > 0), SUM(incoming_count > 0),
               SUM(outgoing_count > 1), SUM(incoming_count > 1),
               MAX(outgoing_count), MAX(incoming_count),
               SUM(outgoing_count > 0 AND incoming_count > 0)
        FROM accounts
    """).fetchone()
    unique_edges = connection.execute("""
        SELECT COUNT(*) FROM (SELECT nameOrig, nameDest FROM transactions GROUP BY nameOrig, nameDest)
    """).fetchone()[0]
    self_loops = connection.execute("SELECT COUNT(*) FROM transactions WHERE nameOrig = nameDest").fetchone()[0]
    unique_self_loops = connection.execute(
        "SELECT COUNT(DISTINCT nameOrig) FROM transactions WHERE nameOrig = nameDest"
    ).fetchone()[0]
    nodes, origins, destinations, repeated_origins, repeated_destinations, max_out, max_in, overlap = map(int, summary)
    for name, table in (("by_step", by_step), ("by_type", by_type), ("by_step_type", by_step_type)):
        table.to_csv(output / f"{name}.csv", index=False)
    distribution = pd.read_sql_query("""
        SELECT outgoing_count, incoming_count, COUNT(*) AS accounts
        FROM accounts GROUP BY outgoing_count, incoming_count ORDER BY outgoing_count, incoming_count
    """, connection)
    distribution.to_csv(output / "account_count_distribution.csv", index=False)
    audit = {
        **basic,
        "duplicates_excluding_first": int(duplicate_rows),
        "fraud": total_fraud,
        "nonfraud": basic["rows"] - total_fraud,
        "fraud_rate": total_fraud / basic["rows"],
        "step_min": int(by_step["step"].min()),
        "step_max": int(by_step["step"].max()),
        "observed_steps": len(by_step),
        "missing_hours_inside_range": int(by_step["step"].max() - by_step["step"].min() + 1 - len(by_step)),
        "unique_origin_accounts": origins,
        "unique_destination_accounts": destinations,
        "unique_accounts_any_role": nodes,
        "accounts_in_both_roles": overlap,
        "origins_with_multiple_transactions": repeated_origins,
        "destinations_with_multiple_transactions": repeated_destinations,
        "max_outgoing_transactions": max_out,
        "max_incoming_transactions": max_in,
        "unique_directed_edges": int(unique_edges),
        "repeated_directed_edges": basic["rows"] - int(unique_edges),
        "self_loop_transactions": int(self_loops),
        "directed_density_excluding_self_loops": (unique_edges - unique_self_loops) / (nodes * (nodes - 1)) if nodes > 1 else 0.0,
        "mean_out_degree_unique": unique_edges / nodes,
    }
    return audit, by_step


def markdown_table(frame: pd.DataFrame) -> str:
    # Avoid adding tabulate solely for a Markdown report.
    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.astype(str).values.tolist()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(output: Path, audit: dict, splits: dict, *, fixture: bool) -> None:
    split_table = pd.DataFrame([
        {"split": name, "step_min": info["step_min"], "step_max": info["step_max"],
         "steps": len(info["steps"]), "rows": info["rows"], "fraud": info["fraud"],
         "fraud_rate": info["fraud"] / info["rows"]}
        for name, info in splits.items()
    ])
    metrics = pd.DataFrame([
        {"metric": key, "value": value} for key, value in audit.items()
        if key not in {"dtypes", "missing"}
    ])
    schema = pd.DataFrame([
        {"column": name, "dtype": audit["dtypes"][name], "missing": audit["missing"][name]}
        for name in COLUMNS
    ])
    type_table = pd.read_csv(output / "by_type.csv")
    text = f"""# Data audit — {'SYNTHETIC TEST FIXTURE; NOT PAYSIM' if fixture else 'USER-SUPPLIED RAW CSV'}

Số liệu dưới đây được tính từ file có SHA-256 trong `dataset_manifest.json`.
Nguồn khai báo: https://www.kaggle.com/datasets/ealaxi/paysim1 .
Checksum xác định file đã xử lý; tự nó không chứng minh file đến từ Kaggle.

## 1. Kích thước, thời gian, tài khoản và cạnh

{markdown_table(metrics)}

## 2. Schema và missing values

{markdown_table(schema)}

Dtype là contract dùng để đọc dữ liệu, không phải dtype suy đoán của CSV.
Duplicates được đếm theo toàn bộ 11 cột đã parse, xuyên chunk; không tự xóa.
Thiếu trường bắt buộc, label không nhị phân, amount âm/vô hạn hoặc step sai thứ tự sẽ làm pipeline dừng.
Balance thiếu được ghi nhận; balance không thuộc feature huấn luyện MVP.

## 3. Phân bố nhãn theo loại giao dịch

{markdown_table(type_table)}

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

{markdown_table(split_table)}

Tỷ lệ áp dụng cho số step quan sát được, dùng floor ở mốc train và train+validation;
không áp dụng cho số dòng. Các giờ không có giao dịch vẫn được báo cáo là gap.
Mọi split phải có cả fraud và nonfraud; nếu không, pipeline dừng và không công bố output.
Không tự dò ranh giới bằng test labels. Thay ranh giới cần quyết định có ghi nhận trước đánh giá.
Mọi E1/E2/E3 dùng chung `split_definition.json` và checksum file nguồn.
Phân bố khác nhau giữa train/validation/test cần được xem xét trước khi train;
không tự cân bằng hoặc chọn lại test để cải thiện metric.

## 6. Point-in-time contract

History(t) = {{x | x.step < t}}. Score toàn bộ step trên state đã đóng băng,
sau đó mới cập nhật state. Không suy ra thứ tự trong giờ từ row_id.
Không dùng label trong history features. Lịch sử giao dịch train được phép warm-up validation,
validation warm-up test; không truyền labels sang feature state.
Fixtures chỉ kiểm tra logic; chưa đủ để kết luận về PaySim hoặc hiệu quả mô hình.
"""
    (output / "data_audit.md").write_text(text, encoding="utf-8")

