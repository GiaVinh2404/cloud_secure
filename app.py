import os
import csv
import torch
import numpy as np
import pandas as pd
import time
from flask import Flask, request, jsonify, render_template_string
from gymnasium.spaces import Discrete
from agents.base_dqn_agent import BaseDQNAgent

# ===== Khởi tạo Flask App =====
app = Flask(__name__)

# ===== Định nghĩa các hằng số =====
LOG_CSV = "network_sim_log.csv"
DATA_PATH = "test/test.parquet"
CHECKPOINT_PATH = "checkpoints/best_combined_checkpoint.pth"

# ===== Khởi tạo CSV log nếu chưa tồn tại =====
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "row_index", "predictor_result", "action_result", "true_label"])

# ===== Hàm ghi log vào CSV =====
def log_to_csv(ts, row_index, predictor_result, action_result, true_label):
    with open(LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, row_index, predictor_result, action_result, true_label])

# ===== Load dataset =====
try:
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c.lower() != "label"]
    obs_size = len(feature_cols)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    MAX_IDX = len(df)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}. Vui lòng kiểm tra lại đường dẫn.")
    df = pd.DataFrame()
    MAX_IDX = 0

# ===== Load pretrained agents =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor_agent = BaseDQNAgent(action_space=Discrete(2), obs_size=obs_size, device=device)
action_agent = BaseDQNAgent(action_space=Discrete(2), obs_size=obs_size + 1, device=device)

try:
    # Tải checkpoint kết hợp đã được lưu từ file train_main.py
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    predictor_agent.load_state_dict(checkpoint["predictor_agent"])
    action_agent.load_state_dict(checkpoint["action_agent"])
    print("✅ Các model đã được tải thành công từ checkpoint kết hợp.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file checkpoint tại {CHECKPOINT_PATH}. Vui lòng chạy train_main.py trước.")
    # Có thể xử lý bằng cách thoát hoặc chạy với model ngẫu nhiên
except Exception as e:
    print(f"Lỗi khi tải checkpoint: {e}")

# ===== API mô phỏng traffic (POST) =====
@app.route("/simulate", methods=["POST"])
def simulate():
    if df.empty:
        return jsonify({"error": "Data not loaded"}), 500
    
    data = request.get_json()
    if not data or "row_index" not in data:
        idx = np.random.randint(0, MAX_IDX)
    else:
        idx = int(data["row_index"])

    if idx < 0 or idx >= MAX_IDX:
        return jsonify({"error": f"row_index out of range [0, {MAX_IDX-1}]"}), 400

    # Predictor Agent
    features = df.iloc[idx][feature_cols].values.astype(np.float32)
    # Lấy hành động mà không cần khám phá (exploitation)
    pred_action = predictor_agent.act(features)

    # Action Agent (nhận thêm kết quả dự đoán)
    action_input = np.concatenate([features, np.array([pred_action], dtype=np.float32)])
    act_action = action_agent.act(action_input)

    # Kết quả
    pred_label = "attack" if pred_action == 1 else "benign"
    act_label = "block" if act_action == 1 else "allow"

    true_label = None
    if label_col:
        true_label_val = int(df.iloc[idx][label_col])
        true_label = "attack" if true_label_val == 1 else "benign"

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    log_to_csv(ts, idx, pred_label, act_label, true_label)

    return jsonify({
        "timestamp": ts,
        "row_index": idx,
        "predictor_result": pred_label,
        "action_result": act_label,
        "true_label": true_label
    })

# ===== API lấy log (GET) =====
@app.route("/log", methods=["GET"])
def get_log():
    if not os.path.exists(LOG_CSV):
        return jsonify([])
    df_log = pd.read_csv(LOG_CSV)
    # Trả về 50 dòng log gần nhất
    return df_log.tail(50).to_dict(orient="records")

# ===== HTML giao diện người dùng =====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Network Simulation Demo</title>
<style>
body { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    margin: 30px; 
    background: #f0f2f5; 
    color: #333;
}
.container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
h1, h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}
p {
    line-height: 1.6;
}
button { 
    padding: 12px 24px; 
    font-size: 16px; 
    background: #3498db; 
    color: white; 
    border: none; 
    border-radius: 8px;
    cursor: pointer; 
    transition: background-color 0.3s;
}
button:hover { 
    background: #2980b9; 
}
#resultBox {
    background: #ecf0f1;
    border: 1px solid #bdc3c7;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
    font-size: 14px;
    line-height: 1.8;
}
#logTable { 
    width: 100%;
    margin-top: 20px; 
    border-collapse: collapse; 
    font-size: 14px;
}
#logTable th, #logTable td { 
    border: 1px solid #e0e0e0; 
    padding: 10px; 
    text-align: left;
}
#logTable th {
    background: #34495e;
    color: white;
    font-weight: bold;
}
#logTable tr:nth-child(even) {
    background: #f8f9fa;
}
</style>
</head>
<body>
<div class="container">
    <h1>Network Traffic Simulation</h1>
    <p>Nhấn <b>Simulate Traffic</b> để mô phỏng một dòng dữ liệu mạng mới đi qua hệ thống.<br>
    Agent sẽ tự động dự đoán và quyết định block/allow. Log quyết định sẽ hiển thị bên dưới.</p>
    <button onclick="simulateTraffic()">Simulate Traffic</button>
    <div id="resultBox"></div>
    <hr style="margin: 30px 0; border: 0; border-top: 1px solid #eee;">
    <h2>Agent Decision Log (latest 50)</h2>
    <table id="logTable"></table>
</div>
<script>
function simulateTraffic() {
    const idx = Math.floor(Math.random() * {{max_idx}});
    fetch("/simulate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({row_index: idx})
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }
        let html = "<b>Timestamp:</b> " + data.timestamp +
                   "<br><b>Row Index:</b> " + data.row_index +
                   "<br><b>Predictor:</b> " + data.predictor_result +
                   "<br><b>Action:</b> " + data.action_result;
        if (data.true_label !== undefined && data.true_label !== null)
            html += "<br><b>True label:</b> " + data.true_label;
        document.getElementById("resultBox").innerHTML = html;
        loadLog();
    })
    .catch(err => alert("Error: " + err));
}
function loadLog() {
    fetch("/log")
    .then(res => res.json())
    .then(rows => {
        let html = "<tr><th>Timestamp</th><th>Row</th><th>Predictor</th><th>Action</th><th>True label</th></tr>";
        for (const row of rows) {
            html += `<tr>
                <td>${row.timestamp}</td>
                <td>${row.row_index}</td>
                <td>${row.predictor_result}</td>
                <td>${row.action_result}</td>
                <td>${row.true_label}</td>
            </tr>`;
        }
        document.getElementById("logTable").innerHTML = html;
    })
    .catch(err => console.error("Could not load log:", err));
}
window.onload = loadLog;
</script>
</body>
</html>
"""

# ===== Route chính để render giao diện =====
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, max_idx=MAX_IDX)

# ===== Chạy ứng dụng =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
