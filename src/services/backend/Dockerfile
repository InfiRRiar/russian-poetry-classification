FROM python:3.11-slim

COPY pyproject.toml uv.lock* README.md* ./
RUN pip install -U pip uv && \ 
    uv sync --frozen --no-dev --group backend --group ml

COPY src/services/backend/ src/services/backend
COPY src/utils.py utils.py

COPY models/onnx models/onnx

CMD ["uv", "run", "uvicorn", "src.services.backend.app:app", "--host", "0.0.0.0", "--port", "8080"]