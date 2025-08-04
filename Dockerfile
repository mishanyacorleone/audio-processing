FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc g++ ffmpeg pkg-config libsndfile1-dev libmagic-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /audio-processing

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

ENV PYTHONPATH=/audio-processing

EXPOSE 8000

CMD ["python", "app/main.py"]