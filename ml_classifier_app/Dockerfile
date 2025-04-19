FROM python:3.8

WORKDIR .
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN mkdir -p /app/models

ENTRYPOINT ["python", "run.py"]