FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    python3-opencv \
    tesseract-ocr \
    libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0"]