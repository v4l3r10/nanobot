FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        sqlite3 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml ./
COPY migrations ./migrations
COPY src ./src
RUN pip install -e .

RUN useradd --uid 10001 --create-home mailbox \
    && mkdir -p /data \
    && chown -R mailbox:mailbox /data /app

USER mailbox

VOLUME ["/data"]
EXPOSE 8765

CMD ["python", "-m", "nanobot_mailbox.server"]
