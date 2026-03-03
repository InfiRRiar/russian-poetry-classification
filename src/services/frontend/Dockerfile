FROM python:3.11-slim

COPY pyproject.toml uv.lock* README.md* ./
RUN pip install -U pip uv && \ 
    uv sync --frozen --no-dev --group frontend

COPY src/services/frontend/ frontend

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "frontend/app.py"]