# Stage 1: download the car-parts dataset from Zenodo and train the model inside the image.
FROM python:3.12-slim AS train
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/download_data.py scripts/
ENV PYTHONPATH=/app/src
RUN python scripts/download_data.py && python -m spareparts.train

# Stage 2: API with the trained model only.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
COPY --from=train /app/models/policy.joblib models/policy.joblib
ENV PYTHONPATH=/app/src:/app MODEL_PATH=/app/models/policy.joblib
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
