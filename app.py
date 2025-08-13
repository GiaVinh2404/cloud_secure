# app.py
import os
import csv
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from gymnasium.spaces import Discrete
from agents.base_dqn_agent import BaseDQNAgent

app = Flask(__name__)

# ==== Đường dẫn file log ====
LOG_CSV = "request_logs.csv"

# ==== Khởi tạo file log nếu chưa có ====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["http_request", "predictor_result", "action_result"])
init_csv()

# ==== Ghi log vào CSV ====
def log_to_csv(http_request, predictor_result, action_result):
    with open(LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([http_request, predictor_result, action_result])

# ==== Hàm tiền xử lý HTTP request thành vector ====
def preprocess_request(http_request: str):
    vector = np.zeros(10, dtype=np.float32)
    req_lower = http_request.lower()

    if "union select" in req_lower or "drop table" in req_lower:
        vector[0] = 1  # SQLi
    elif "<script>" in req_lower:
        vector[1] = 1  # XSS
    elif "flood" in req_lower or "dos" in req_lower:
        vector[2] = 1  # DoS
    else:
        vector[9] = 1  # benign

    vector += np.random.rand(10) * 0.05
    return vector.astype(np.float32)

# ==== Load model đã train ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor_agent = BaseDQNAgent(
    action_space=Discrete(2),  # attack / benign
    obs_size=10,
    device=device
)
action_agent = BaseDQNAgent(
    action_space=Discrete(2),  # block / allow
    obs_size=11,
    device=device
)

predictor_agent.load("checkpoints/predictor_agent_best.pth")
action_agent.load("checkpoints/action_agent_best.pth")
print("✅ Models loaded successfully.")

# ==== API nhận request JSON ====
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "http_request" not in data:
        return jsonify({"error": "Missing 'http_request' field"}), 400

    http_request_text = data["http_request"]

    predictor_obs = preprocess_request(http_request_text)
    pred_action = predictor_agent.act(predictor_obs)

    action_obs = np.concatenate([predictor_obs, np.array([pred_action], dtype=np.float32)])
    act_action = action_agent.act(action_obs)

    pred_label = "attack" if pred_action == 1 else "benign"
    act_label = "block" if act_action == 1 else "allow"

    log_to_csv(http_request_text, pred_label, act_label)

    return jsonify({
        "predictor_result": pred_label,
        "action_result": act_label
    })

# ==== Giao diện HTML để test ====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cloud Security RL - HTTP Request Tester</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background: #f9f9f9; }
        h1 { color: #333; }
        textarea { width: 100%; height: 120px; padding: 10px; font-size: 14px; }
        button { padding: 10px 20px; font-size: 16px; background: #007BFF; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 10px; background: #fff; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>HTTP Request Security Checker</h1>
    <textarea id="requestInput" placeholder="Nhập HTTP request ở đây..."></textarea><br><br>
    <button onclick="sendRequest()">Kiểm tra</button>
    <div class="result" id="resultBox"></div>

    <script>
        function sendRequest() {
            const reqText = document.getElementById("requestInput").value;
            fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({http_request: reqText})
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("resultBox").innerHTML =
                    "<b>Predictor:</b> " + data.predictor_result +
                    "<br><b>Action:</b> " + data.action_result;
            })
            .catch(err => alert("Error: " + err));
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
