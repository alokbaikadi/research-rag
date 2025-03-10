FROM python:3.12-slim
WORKDIR /app

ENV LLM_MODEL='qwen2.5'
ENV EMBEDDING_MODEL='all-minilm'

RUN apt update
RUN apt install -y curl
RUN mkdir /data
RUN mkdir /models

ENV OLLAMA_MODELS=/models

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN cat > .env <<EOF
LLM_MODEL="\$LLM_MODEL"
PERSIST_DIRECTORY="chroma_langchain.db"
EMBEDDING_MODEL="=\$EMBEDDING_MODEL"
DATA_DIR="/data"
EOF

COPY src/*.py .
COPY pyproject.toml .
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh


RUN pip install poetry
RUN python -m venv .venv
RUN poetry install
