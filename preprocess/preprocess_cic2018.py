"""
Tiền xử lý dữ liệu CSE-CIC-IDS2018:
- Đọc tất cả file CSV trong data/raw
- Chuẩn hóa tên cột
- Loại bỏ cột không cần
- Chuẩn hóa label
- Loại bỏ NaN, giá trị vô cực, trùng lặp
- Ép kiểu an toàn trước khi lưu
- Lưu sang định dạng Parquet để train nhanh hơn
"""

import os
import numpy as np
import pandas as pd
from fastai.tabular.all import df_shrink
from fastcore.parallel import parallel

# ==========================
# Cấu hình thư mục
# ==========================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ==========================
# Mapping tên cột
# ==========================
col_name_consistency = {
    'Flow ID': 'Flow ID',
    'Source IP': 'Source IP',
    'Src IP': 'Source IP',
    'Source Port': 'Source Port',
    'Src Port': 'Source Port',
    'Destination IP': 'Destination IP',
    'Dst IP': 'Destination IP',
    'Destination Port': 'Destination Port',
    'Dst Port': 'Destination Port',
    'Protocol': 'Protocol',
    'Timestamp': 'Timestamp',
    'Flow Duration': 'Flow Duration',
    'Total Fwd Packets': 'Total Fwd Packets',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Total Backward Packets': 'Total Backward Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'Total Length of Fwd Packets': 'Fwd Packets Length Total',
    'TotLen Fwd Pkts': 'Fwd Packets Length Total',
    'Total Length of Bwd Packets': 'Bwd Packets Length Total',
    'TotLen Bwd Pkts': 'Bwd Packets Length Total',
    'Fwd Packet Length Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Packet Length Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Packet Length Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Packet Length Std': 'Fwd Packet Length Std',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Packet Length Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Packet Length Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Packet Length Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Packet Length Std': 'Bwd Packet Length Std',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Bytes/s': 'Flow Bytes/s',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Packets/s': 'Flow Packets/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Total',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Label': 'Label'
}

drop_columns = [
    "Flow ID",
    'Fwd Header Length.1',
    "Source IP", "Src IP",
    "Source Port", "Src Port",
    "Destination IP", "Dst IP",
    "Destination Port", "Dst Port",
    "Timestamp"
]

# ==========================
# Hàm xử lý từng file
# ==========================
def process_csv(path):
    print(f"📂 Đang xử lý: {path}")
    df = pd.read_csv(path, sep=",", encoding="utf-8")

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip()
    df.rename(columns=col_name_consistency, inplace=True)
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Chuẩn hóa label
    df['Label'] = df['Label'].replace({'BENIGN': 'Benign'})

    # Fix lỗi pyarrow: ép toàn bộ category -> string
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype(str)

    # Xử lý cột Protocol: giữ dạng string hoặc int an toàn
    if 'Protocol' in df.columns:
        try:
            df['Protocol'] = pd.to_numeric(df['Protocol'], errors='raise').astype(np.int32)
        except ValueError:
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

# ==========================
# Chạy xử lý toàn bộ
# ==========================
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
        print("🎯 Hoàn thành xử lý tất cả file.")
