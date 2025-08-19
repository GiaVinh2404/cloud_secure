import os
import pandas as pd

input_folder = "data"
output_file = "merged.csv"

all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
all_files.sort()

first_file = True
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    for file in all_files:
        file_path = os.path.join(input_folder, file)
        print(f"Đang đọc file: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        
        if first_file:
            df.to_csv(outfile, index=False)
            first_file = False
        else:
            df.to_csv(outfile, index=False, header=False)

print(f"✅ Đã gộp xong {len(all_files)} file vào '{output_file}'")
