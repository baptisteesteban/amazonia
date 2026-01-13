FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN apt update -y && \
    apt install -y --no-install-recommends \
    cmake \
    ninja-build \
    gcc \
    g++ \
    gdb \
    clangd \
    curl \
    wget \
    git \
    doxygen \
    doxygen-doc && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu

RUN curl -LsSf https://astral.sh/uv/install.sh | sh