"""
Tiền xử lý dữ liệu CSE-CIC-IDS2018:
- Đọc tất cả file CSV trong data/raw
- Chuẩn hóa tên cột
- Loại bỏ cột không cần
- Chuẩn hóa label (0: Benign, 1: Attack)
- Loại bỏ NaN, giá trị vô cực, trùng lặp
- Cân bằng dữ liệu (nếu cần)
- Ép kiểu an toàn trước khi lưu
- Lưu sang định dạng Parquet để train nhanh hơn
"""

import os
import numpy as np
import pandas as pd
from fastai.tabular.all import df_shrink
from fastcore.parallel import parallel

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

col_name_consistency = {
    # ...giữ nguyên như bạn đã mapping...
    # (bạn có thể copy lại phần mapping cột ở trên)
}

drop_columns = [
    "Flow ID", 'Fwd Header Length.1',
    "Source IP", "Src IP", "Source Port", "Src Port",
    "Destination IP", "Dst IP", "Destination Port", "Dst Port",
    "Timestamp"
]

def normalize_label(label):
    """Chuyển label về 0 (Benign) và 1 (Attack)"""
    if str(label).strip().lower() in ["benign", "benign\n", "0", "normal"]:
        return 0
    return 1

def process_csv(path):
    print(f"📂 Đang xử lý: {path}")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    df.columns = df.columns.str.strip()
    df.rename(columns=col_name_consistency, inplace=True)
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Chuẩn hóa label
    df['Label'] = df['Label'].replace({'BENIGN': 'Benign'})
    df['Label'] = df['Label'].apply(normalize_label)

    # Ép toàn bộ các cột (trừ Label) về float nếu có thể, nếu không thì về string
    for col in df.columns:
        if col != 'Label':
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception:
                df[col] = df[col].astype(str)

    # Xử lý cột Protocol: giữ dạng string hoặc int an toàn
    if 'Protocol' in df.columns:
        try:
            df['Protocol'] = pd.to_numeric(df['Protocol'], errors='raise').astype(np.int32)
        except Exception:
            df['Protocol'] = df['Protocol'].astype(str)

    # Loại bỏ giá trị vô cực và NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Xóa trùng lặp
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Thu gọn dtype
    df = df_shrink(df)

    # Lưu parquet
    filename = os.path.basename(path).replace(".csv", ".parquet")
    save_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_parquet(save_path, index=False, engine="pyarrow")
    print(f"✅ Lưu xong: {save_path}")
    return save_path

if __name__ == "__main__":
    csv_files = [
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print("⚠ Không tìm thấy file CSV trong thư mục data/raw")
    else:
        processed_paths = parallel(process_csv, csv_files, progress=True)
        # Gộp lại thành 1 file parquet lớn và cân bằng dữ liệu nếu cần
        dfs = [pd.read_parquet(p) for p in processed_paths]
        df_all = pd.concat(dfs, ignore_index=True)
        # Cân bằng dữ liệu (optional, nếu dữ liệu lệch nhiều)
        min_count = min(df_all["Label"].value_counts().values)
        df_balanced = pd.concat([
            df_all[df_all["Label"] == 0].sample(min_count, random_state=42),
            df_all[df_all["Label"] == 1].sample(min_count, random_state=42)
        ], ignore_index=True)
        df_balanced = df_shrink(df_balanced)
        final_path = os.path.join(PROCESSED_DATA_DIR, "cic2018_processed.parquet")
        df_balanced.to_parquet(final_path, index=False)
        print(f"🎯 Đã lưu file tổng hợp cân bằng: {final_path}")