FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 absa
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

USER 1000:1000
ENTRYPOINT ["python", "-m", "absa"]
CMD ["--help"]
