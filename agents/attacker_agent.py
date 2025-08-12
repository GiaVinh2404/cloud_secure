import numpy as np
import torch
from .base_dqn_agent import BaseDQNAgent
import random
import string

class AttackAgent(BaseDQNAgent):
    def __init__(self, action_space, obs_size, attack_types, device=None, *args, **kwargs):
        super().__init__(action_space, obs_size, device=device, *args, **kwargs)
        self.attack_types = attack_types  # Ví dụ: ["normal", "SQLi", "XSS", "RCE"]

    def generate_random_string(self, length=6):
        # Sinh chuỗi ký tự ngẫu nhiên cho biến thể payload
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(random.choice(chars) for _ in range(length))

    def generate_request(self, attack_type):
        # Sinh request đa dạng hơn dựa trên kiểu tấn công và random payload
        if attack_type == "normal":
            # request bình thường, cố định hoặc có biến thể nhỏ
            page = random.choice(["index.html", "home", "about", "contact"])
            return f"GET /{page}.php HTTP/1.1\r\nHost: example.com\r\n\r\n"

        elif attack_type == "SQLi":
            payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT password FROM users WHERE '1'='1",
                f"' OR username='{self.generate_random_string()}' --"
            ]
            payload = random.choice(payloads)
            return f"GET /search.php?q={payload} HTTP/1.1\r\nHost: example.com\r\n\r\n"

        elif attack_type == "XSS":
            scripts = [
                "<script>alert('XSS')</script>",
                f"<img src=x onerror=alert('{self.generate_random_string()}')>",
                "<svg/onload=alert('XSS')>",
                f"<body onload=alert('{self.generate_random_string()}')>"
            ]
            script = random.choice(scripts)
            return f"GET /comment.php?msg={script} HTTP/1.1\r\nHost: example.com\r\n\r\n"

        elif attack_type == "RCE":
            commands = [
                "whoami",
                "ls -la",
                "cat /etc/passwd",
                f"echo {self.generate_random_string()}"
            ]
            cmd = random.choice(commands)
            return f"GET /vulnerable.php?cmd={cmd} HTTP/1.1\r\nHost: example.com\r\n\r\n"

        else:
            return "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"

    def encode_request(self, request):
        max_len = 256
        vector = np.zeros(max_len, dtype=np.float32)
        for i, c in enumerate(request[:max_len]):
            vector[i] = ord(c) / 256.0
        return vector

    def act(self, observation):
        # Lựa chọn action theo epsilon-greedy
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.attack_types))
        else:
            obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(obs_t)
            idx = int(torch.argmax(q_values).item())

        attack_type = self.attack_types[idx]
        request = self.generate_request(attack_type)
        request_vector = self.encode_request(request)

        # Trả về action index và vector request mã hóa
        return idx, request_vector
