# Sử dụng Python image
FROM python:3.10-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ code vào container
COPY . .

# Cài đặt thư viện Python từ requirements.txt (nếu có)
RUN pip install --no-cache-dir -r requirements.txt

# Mở port cho Flask (ví dụ 5000)
EXPOSE 5000

# Chạy ứng dụng
CMD ["python", "app.py"]
