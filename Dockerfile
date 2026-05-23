# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install Node.js 20 for the WhatsApp bridge
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg git bubblewrap openssh-client && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y gnupg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# v0.2.0 added a custom hatchling build hook (hatch_build.py, the
# webui-build plugin). hatchling instantiates the hook class during ANY
# `pip install .` — including the cached deps-only layer below — and
# raises "Build script does not exist: hatch_build.py" if the file is
# absent from the build context, before the hook's own skip logic runs.
# Upstream's Dockerfile never copies webui/ nor the prebuilt
# nanobot/web/dist/, so the hook is meant to no-op here: make that
# explicit and deterministic with the hook's first-class bypass instead
# of relying on the incidental "no webui/ source tree" branch. Bronzo is
# Telegram-first; the bundled web UI is not a served surface.
ENV NANOBOT_SKIP_WEBUI_BUILD=1

# Install Python dependencies first (cached layer)
# hatch_build.py must be present at `uv pip install .` time: the custom
# hatchling build hook is loaded during the build and errors with
# "Build script does not exist: hatch_build.py" if it is missing here.
COPY pyproject.toml README.md LICENSE hatch_build.py ./
RUN mkdir -p nanobot bridge && touch nanobot/__init__.py && \
    uv pip install --system --no-cache '.[wiki-search]' && \
    rm -rf nanobot bridge

# Copy the full source and install
COPY nanobot/ nanobot/
COPY bridge/ bridge/
RUN uv pip install --system --no-cache '.[wiki-search]'

# Build the WhatsApp bridge
WORKDIR /app/bridge
RUN git config --global --add url."https://github.com/".insteadOf ssh://git@github.com/ && \
    git config --global --add url."https://github.com/".insteadOf git@github.com: && \
    npm install && npm run build
WORKDIR /app

# Create non-root user and config directory
RUN useradd -m -u 1000 -s /bin/bash nanobot && \
    mkdir -p /home/nanobot/.nanobot && \
    chown -R nanobot:nanobot /home/nanobot /app

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

USER nanobot
ENV HOME=/home/nanobot

# Pre-bake the wiki-search embedding model into the image so the Dream cycle
# never downloads it at runtime. fastembed stores models under
# FASTEMBED_CACHE_PATH (it IGNORES HF_HOME), default /tmp/fastembed_cache —
# pin it to a stable path under $HOME/.cache, which is NOT a mounted volume
# (only ~/.nanobot is), so the baked layer is exactly what the runtime reads
# and the volume mount can't shadow it. Pinned to the DreamConfig default;
# override to match a custom wiki_embedding_model with
#   --build-arg WIKI_EMBEDDING_MODEL=<id>
# Granite is a public model and downloads fine unauthenticated; an HF token is
# OPTIONAL (only raises HF Hub rate limits / silences the unauthenticated
# warning). Pass it as a BuildKit secret so it is never recorded in ENV or
# `docker history`:
#   docker build --secret id=hf_token,env=HF_TOKEN ...        (from host env)
#   docker build --secret id=hf_token,src=./hf_token.txt ...  (from a file)
# FAIL-FAST: the build EXITS NON-ZERO if the model can't be baked, so an image
# that would otherwise download the model at runtime is never shipped. Retry
# the build if a transient HF outage causes a miss.
ENV FASTEMBED_CACHE_PATH=/home/nanobot/.cache/fastembed
ENV HF_HOME=/home/nanobot/.cache/huggingface
ARG WIKI_EMBEDDING_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
RUN --mount=type=secret,id=hf_token,uid=1000,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -c "from nanobot.agent.wiki.embeddings import warm_embedding_model as w; import sys; sys.exit(0 if w('${WIKI_EMBEDDING_MODEL}') else 1)"

# Gateway default port
EXPOSE 18790

ENTRYPOINT ["entrypoint.sh"]
CMD ["status"]
