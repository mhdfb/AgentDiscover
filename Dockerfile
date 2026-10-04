# AgentDiscover — the environment every search agent and evaluator runs in.
# Built in CI (.github/workflows/image.yml), pushed to ghcr.io, and run with any runtime:
#
#   singularity exec docker://ghcr.io/<you>/agentdiscover:v1 ...   # Singularity/Apptainer
#   srun --container-image=ghcr.io/<you>/agentdiscover:v1 ...      # Enroot + Pyxis
#   docker run ghcr.io/<you>/agentdiscover:v1 ...                  # laptop
#
# No secrets in this file (pass them at launch with --env), and only what is identical
# for every agent: worktrees, guidance and solution.py are mounted at run time.

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git, curl, procps; g++ and the AtCoder judge's libraries for compiling C++ tasks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates procps \
        g++ libboost-dev libeigen3-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# ac-library (header-only) at the judge's path; v1.5.1 is the judge's version.
RUN git clone --depth 1 --branch v1.5.1 https://github.com/atcoder/ac-library.git \
        /opt/ac-library && rm -rf /opt/ac-library/.git

# Node 24: the coding-agent CLIs are npm packages.
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The coding agents, pinned so a run is reproducible. Bump deliberately.
RUN npm install -g --no-fund --no-audit \
        @anthropic-ai/claude-code@2.1.287 \
        @openai/codex@0.159.2 \
    && npm cache clean --force

# Python packages: neo4j for the graph database; numpy, scipy and PyWavelets so an agent
# can run its own solution.py before submitting it.
RUN pip install --no-cache-dir \
        neo4j==5.28.1 \
        numpy==2.5.2 \
        scipy==1.18.1 \
        PyWavelets==1.9.0

# A neutral working directory. Every runtime mounts the real worktree over /work.
WORKDIR /work

# No ENTRYPOINT on purpose: Singularity and Enroot handle entrypoints differently.
CMD ["/bin/bash"]
