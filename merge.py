import os
import pandas as pd

# Thư mục chứa các file CSV
input_folder = "data"
# File CSV đầu ra
output_file = "merged.csv"

# Lấy danh sách tất cả file CSV trong thư mục
all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
all_files.sort()  # Sắp xếp theo tên file

# Gộp file
first_file = True
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    for file in all_files:
        file_path = os.path.join(input_folder, file)
        print(f"Đang đọc file: {file_path}")
        
        # Đọc CSV
        df = pd.read_csv(file_path, low_memory=False)
        
        # Nếu là file đầu tiên → ghi cả header
        if first_file:
            df.to_csv(outfile, index=False)
            first_file = False
        else:
            # Các file sau → ghi không có header
            df.to_csv(outfile, index=False, header=False)

print(f"✅ Đã gộp xong {len(all_files)} file vào '{output_file}'")
