FROM public.ecr.aws/deep-learning-containers/base:13.0.0-gpu-py312-cu130-ubuntu22.04-ec2

# jq: the entrypoint (accelerate_sagemaker_train.sh) parses $SM_HOSTS with it to compute
# num_machines / machine_rank for `accelerate launch`. The DLC base does NOT ship jq.
RUN apt-get update && apt-get install -y --no-install-recommends jq \
    && rm -rf /var/lib/apt/lists/*

# Install UV.
RUN python -m pip install uv
# Create brand new env.
RUN uv venv /opt/venv
# Set environment variables to "activate" the venv for all subsequent steps
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# UV_HTTP_TIMEOUT: NVIDIA's wheel index (pypi.nvidia.com) is slow for the big CUDA wheels
#   (cublas/nccl/cusolver are 100-500MB); the default ~120s timeout kills them. 5 min lets
#   them finish. UV_CONCURRENT_DOWNLOADS caps parallel fetches so each big wheel gets more
#   bandwidth instead of ~40 streams fighting one CDN (the real cause of the timeouts).
ENV UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4
COPY requirements.txt ./requirements.txt
RUN uv pip install --no-cache -r requirements.txt \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130

# FlashAttention
RUN uv pip install --no-cache flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2141
# Pinned to b30: b35 (latest) crashes in the varlen split-KV path on packed CPT sequences
#   ("Expected IntTuple ... got <seqlen>"); b30 is the known-good beta for this stack.
# pinned to b30 since the varlen split-KV path on packed CPT sequences yielded error: ("Expected IntTuple ... got <seqlen>")
RUN uv pip install --no-build-isolation "flash-attn-4[cu13]==4.0.0b30"

# MAX_JOBS=2 caps peak RAM in the 7.7GB VM.
RUN MAX_JOBS=2 \
    uv pip install --no-build-isolation \
    flash-linear-attention==0.5.2 \
    causal-conv1d==1.7.0

# The DLC base ships a system cuDNN that shadows torch's bundled one on LD_LIBRARY_PATH,
# triggering "cuDNN version incompatibility" at runtime. Put torch's bundled nvidia libs
# FIRST so the version torch was built against is the one that loads.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# This is a sanity check to ensure the fragile GPU kernels import against the real torch.
RUN python -c "import sys, torch, transformers, trl, peft, deepspeed, flash_attn_3, flash_attn.cute, causal_conv1d, fla; \
print('python   :', sys.executable); \
print('torch    :', torch.__version__); \
print('cuda     :', torch.version.cuda); \
print('cudnn    :', torch.backends.cudnn.version()); \
print('transformers:', transformers.__version__); \
assert '+cu' in torch.__version__, 'torch is not a CUDA build'; \
print('fa3 / fa4 / causal_conv1d / fla all import OK')"
