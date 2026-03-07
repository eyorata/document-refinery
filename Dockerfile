FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY rubric ./rubric
COPY streamlit_app.py ./
COPY DOMAIN_NOTES.md ./
COPY .env.example ./

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8501

# Default to Streamlit UI. Override at runtime for CLI mode.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

