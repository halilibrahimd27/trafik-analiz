# syntax=docker/dockerfile:1
# Trafik İşareti Tanıma Sistemi — Streamlit + TensorFlow (CNN v3)
FROM python:3.11-slim

# OpenCV (CLAHE) için gerekli sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Bağımlılıkları önce kur (Docker katman önbelleği için)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Uygulama kaynakları (veri hariç — compose ile salt-okunur mount edilir)
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY results/ ./results/
COPY samples/ ./samples/
COPY models/trafik_model.keras ./models/trafik_model.keras
COPY README.md ./README.md

EXPOSE 8501

# Streamlit sağlık ucu ile container sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
