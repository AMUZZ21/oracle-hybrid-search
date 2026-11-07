# Hybrid Search Lab: web UI + API.
#   docker build -t hybrid-search .                          # demo corpus
#   docker build -t hybrid-search --build-arg CORPUS=scifact .
#   docker run -p 7860:7860 hybrid-search                    # then open http://localhost:7860
FROM python:3.11-slim

RUN useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.hf

WORKDIR /app
COPY --chown=user requirements.txt .
# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

USER user
COPY --chown=user . /app

ARG CORPUS=demo
ENV HS_CORPUS=$CORPUS

# Bake models, dataset and embeddings into the image so the first request is fast.
RUN python -m webapp.warmup

EXPOSE 7860
CMD ["sh", "-c", "uvicorn webapp.server:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]
