FROM python:3.10-slim

# Đặt thư mục làm việc
WORKDIR /app

# Copy file chính và requirements
COPY app.py .
COPY requirements.txt .

# Copy các thư mục cần thiết cho code
COPY agents/ ./agents/
COPY env/ ./env/

# (Tùy chọn) copy checkpoint và dataset
COPY checkpoints/ ./checkpoints/
COPY test/ ./test/
COPY config/ ./config/

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Mở port Flask
EXPOSE 5000

# Chạy Flask
CMD ["python", "app.py"]
