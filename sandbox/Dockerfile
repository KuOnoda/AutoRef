# The proposer's container. It holds no code of its own: the Claude Code binary, the virtualenv and
# the repository files the proposer may read are bind-mounted in by sandbox/claude.
#
#   docker build -t autoref-proposer:latest sandbox/
#
# ubuntu:24.04 carries glibc 2.39, which the native claude binary needs, and python 3.12, which a
# virtualenv built on the host's system python 3.12 expects to find at /usr/bin/python3 (a venv built
# on a uv/pyenv/conda python has its interpreter mounted in instead). build-essential and python3-dev
# are there because triton compiles its CUDA driver shim on first use.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ripgrep git less python3 libglib2.0-0 libgl1 \
        build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1
