# Gunakan image dasar Python
FROM python:3.8-slim

# Set direktori kerja dalam container
WORKDIR /app

# Salin file dari direktori lokal ke dalam container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tentukan port yang digunakan oleh aplikasi
EXPOSE 8080

# Perintah untuk menjalankan aplikasi
CMD ["python", "app.py"]
