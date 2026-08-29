FROM python:3.12-slim-bookworm
LABEL org.opencontainers.image.source="https://github.com/cisgroup/CEE420-Urban-AI"
LABEL org.opencontainers.image.description="CEE420 Urban AI course environment (Princeton, Fall 2026)"

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLCONFIGDIR=/opt/mplcache \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 -s /bin/bash student \
 && mkdir -p /opt/mplcache && chmod 777 /opt/mplcache

# Dependencies come ONLY from the two source-of-truth files; --locked fails on a stale lock.
COPY pyproject.toml uv.lock /opt/project/
WORKDIR /opt/project
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# One obvious kernel in the VS Code picker; pre-warm the matplotlib font cache.
RUN /opt/venv/bin/python -m ipykernel install \
      --name urban-ai --display-name "Urban AI (Python 3.12)" \
 && /opt/venv/bin/python -c "import matplotlib.pyplot"

USER student
WORKDIR /workspaces
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
