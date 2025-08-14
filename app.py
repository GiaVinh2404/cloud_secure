import os
import csv
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from env.cloud_security_env import CloudSecurityEnv
from agents.base_dqn_agent import BaseDQNAgent

# ===== Flask App =====
app = Flask(__name__)
LOG_CSV = "request_logs.csv"

# ===== Init CSV log =====
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["http_request", "predictor_result", "action_result"])

def log_to_csv(http_request, predictor_result, action_result):
    with open(LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([http_request, predictor_result, action_result])

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load env để lấy đúng obs_size =====
env = CloudSecurityEnv(
    dataset_path="data/processed/cic2018_processed.parquet",
    max_steps=1  # chỉ cần để lấy shapes
)

# ===== Khởi tạo agents đúng kiến trúc =====
predictor_agent = BaseDQNAgent(
    action_space=env.action_spaces["predictor_agent"],
    obs_size=env.observation_spaces["predictor_agent"].shape[0],
    device=device
)

action_agent = BaseDQNAgent(
    action_space=env.action_spaces["action_agent"],
    obs_size=env.observation_spaces["action_agent"].shape[0],
    device=device
)

# ===== Load checkpoints =====
predictor_agent.load("checkpoints/predictor_agent_best.pth")
action_agent.load("checkpoints/action_agent_best.pth")
print("✅ Models loaded successfully.")

# ===== Chuyển HTTP request thành feature vector 77 chiều =====
def http_request_to_features_77(http_request: str):
    """
    Mapping đơn giản minh họa: 77 chiều, 1 vài index đặc biệt đánh dấu loại tấn công.
    Trong thực tế cần map đúng preprocessing lúc train.
    """
    vector = np.zeros(env.observation_spaces["predictor_agent"].shape[0], dtype=np.float32)
    req_lower = http_request.lower()

    # Ví dụ mapping cơ bản
    if "union select" in req_lower or "drop table" in req_lower:
        vector[0] = 1.0   # SQL Injection
    elif "<script>" in req_lower or "javascript:" in req_lower:
        vector[1] = 1.0   # XSS
    elif "flood" in req_lower or "dos" in req_lower:
        vector[2] = 1.0   # DoS
    else:
        vector[-1] = 1.0  # benign

    # Noise nhẹ để gần với phân phối train
    vector += np.random.rand(vector.shape[0]) * 0.05
    return vector.astype(np.float32)

# ===== API nhận request JSON =====
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "http_request" not in data:
        return jsonify({"error": "Missing 'http_request' field"}), 400

    http_request_text = data["http_request"]
    features = http_request_to_features_77(http_request_text)

    # Predictor agent
    pred_action = predictor_agent.act(features)

    # Action agent
    action_input = np.concatenate([features, np.array([pred_action], dtype=np.float32)])
    act_action = action_agent.act(action_input)

    pred_label = "attack" if pred_action == 1 else "benign"
    act_label = "block" if act_action == 1 else "allow"

    log_to_csv(http_request_text, pred_label, act_label)

    return jsonify({
        "predictor_result": pred_label,
        "action_result": act_label
    })

# ===== Web UI =====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Cloud Security RL - HTTP Request Tester</title>
<style>
body { font-family: Arial; margin: 30px; background: #f9f9f9; }
textarea { width: 100%; height: 120px; padding: 10px; font-size: 14px; }
button { padding: 10px 20px; font-size: 16px; background: #007BFF; color: white; border: none; cursor: pointer; }
button:hover { background: #0056b3; }
.result { margin-top: 20px; padding: 10px; background: #fff; border: 1px solid #ccc; }
</style>
</head>
<body>
<h1>HTTP Request Security Checker</h1>
<textarea id="requestInput" placeholder="Nhập HTTP request ở đây..."></textarea><br><br>
<button onclick="sendRequest()">Check Request</button>
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

# ===== Main =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
