"""
Tiền xử lý dữ liệu CSE-CIC-IDS2018:
- Đọc file merged.csv ở thư mục gốc dự án theo từng chunk nhỏ (tránh hết RAM)
- Nếu dòng nào lỗi format sẽ tự động bỏ qua, không tiền xử lý dòng đó
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

MERGED_CSV_PATH = "merged.csv"
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
    if str(label).strip().lower() in ["benign", "benign\n", "0", "normal"]:
        return 0
    return 1

def process_csv_in_chunks(path, chunk_size=200_000):
    chunk_paths = []
    chunk_iter = pd.read_csv(
        path, sep=",", encoding="utf-8", on_bad_lines='skip', low_memory=False, chunksize=chunk_size
    )
    for i, chunk in enumerate(chunk_iter):
        chunk.columns = chunk.columns.str.strip()
        chunk.rename(columns=col_name_consistency, inplace=True)
        chunk.drop(columns=drop_columns, inplace=True, errors="ignore")
        chunk['Label'] = chunk['Label'].replace({'BENIGN': 'Benign'})
        chunk['Label'] = chunk['Label'].apply(normalize_label)
        for col in chunk.columns:
            if col != 'Label':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        if 'Protocol' in chunk.columns:
            chunk['Protocol'] = pd.to_numeric(chunk['Protocol'], errors='coerce').astype('Int32')
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)
        chunk.drop_duplicates(inplace=True)
        chunk.reset_index(drop=True, inplace=True)
        chunk = df_shrink(chunk)
        chunk_path = os.path.join(PROCESSED_DATA_DIR, f"chunk_{i}.parquet")
        chunk.to_parquet(chunk_path, index=False, engine="pyarrow")
        print(f"✅ Đã lưu chunk: {chunk_path}")
        chunk_paths.append(chunk_path)
    return chunk_paths

if __name__ == "__main__":
    if not os.path.exists(MERGED_CSV_PATH):
        print("⚠ Không tìm thấy file merged.csv ở thư mục gốc dự án")
    else:
        chunk_paths = process_csv_in_chunks(MERGED_CSV_PATH, chunk_size=200_000)
        # Gộp lại thành 1 file parquet lớn và cân bằng dữ liệu nếu cần
        dfs = [pd.read_parquet(p) for p in chunk_paths]
        df_all = pd.concat(dfs, ignore_index=True)
        min_count = min(df_all["Label"].value_counts().values)
        df_balanced = pd.concat([
            df_all[df_all["Label"] == 0].sample(min_count, random_state=42),
            df_all[df_all["Label"] == 1].sample(min_count, random_state=42)
        ], ignore_index=True)
        df_balanced = df_shrink(df_balanced)
        final_path = os.path.join(PROCESSED_DATA_DIR, "cic2018_processed.parquet")
        df_balanced.to_parquet(final_path, index=False)
        print(f"🎯 Đã lưu file tổng hợp cân bằng: {final_path}")