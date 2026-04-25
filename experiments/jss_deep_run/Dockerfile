FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ca-certificates \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    biber \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

RUN pip install --upgrade pip && \
    pip install "jax[cuda12_pip]==0.4.30" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Build context is the repository root:
#   docker build -f experiments/jss_deep_run/Dockerfile -t econirl-deep-run:v1 .
WORKDIR /workspace/econirl
COPY . /workspace/econirl
RUN pip install -e .[dev]

# Generic bash entrypoint so dispatch_runpod.py and cloud_test.py can
# both pass full command strings as dockerArgs. Matrix runs invoke
# `python -m experiments.jss_deep_run.run_cell ...`; cloud_test runs
# arbitrary pytest selectors or shell commands.
ENTRYPOINT ["/bin/bash", "-lc"]
