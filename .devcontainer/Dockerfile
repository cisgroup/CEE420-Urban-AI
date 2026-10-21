FROM python:3.12-slim-bookworm
LABEL org.opencontainers.image.source="https://github.com/cisgroup/CEE420-Urban-AI"
LABEL org.opencontainers.image.description="CEE420 Urban AI course environment (Princeton, Fall 2026)"

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MPLCONFIGDIR=/opt/mplcache \
    TORCH_HOME=/opt/torch \
    HF_HOME=/opt/hf \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 -s /bin/bash student \
 && mkdir -p /opt/mplcache && chmod 777 /opt/mplcache

# Debian's /etc/profile overwrites PATH rather than appending to it, so a LOGIN shell
# (bash -l, or a terminal profile configured with -l) throws away the /opt/venv/bin set
# above and finds the system python, which has none of the course packages. VS Code's
# default terminal is not a login shell and is unaffected, but anyone who opens one gets
# an inexplicable ModuleNotFoundError. profile.d is sourced after that reset, so putting
# the venv back here covers both kinds of shell.
RUN printf 'PATH=/opt/venv/bin:$PATH\n' > /etc/profile.d/10-venv-path.sh

# Dependencies come ONLY from the two source-of-truth files; --locked fails on a stale lock.
COPY pyproject.toml uv.lock /opt/project/
WORKDIR /opt/project
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

# One obvious kernel in the VS Code picker; pre-warm the matplotlib font cache; import torch once so a broken
# wheel fails the build and not a student; fetch the ImageNet weights of ResNet-18 (45 MB) into the shared torch
# cache, so 04_03 and the P05 kit run offline inside the container; and fetch the open weights of the P06 kit, SAM 2.1
# hiera-tiny (156 MB) and CLIP ViT-B/16 (600 MB), into the shared Hugging Face cache for the same reason. The hub holds CLIP's
# weights in the old binary form on its main revision and a converted copy on a side branch, and the loader fetches both; the
# offline loader resolves the main revision, so the side copy is dropped and the image stays 600 MB smaller.
RUN /opt/venv/bin/python -m ipykernel install \
      --name urban-ai --display-name "Urban AI (Python 3.12)" \
 && /opt/venv/bin/python -c "import matplotlib.pyplot" \
 && /opt/venv/bin/python -c "import torch, torchvision; from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); print(torch.__version__, torchvision.__version__)" \
 && /opt/venv/bin/python -c "import transformers; from transformers import CLIPModel, CLIPProcessor, Sam2Model, Sam2Processor; [f.from_pretrained(m) for m, f in (('facebook/sam2.1-hiera-tiny', Sam2Model), ('facebook/sam2.1-hiera-tiny', Sam2Processor), ('openai/clip-vit-base-patch16', CLIPModel), ('openai/clip-vit-base-patch16', CLIPProcessor))]; print(transformers.__version__)" \
 && for f in $(find /opt/hf/hub -path '*clip-vit-base-patch16*' -name model.safetensors); do rm -f "$(readlink -f "$f")" "$f"; done \
 && chmod -R a+rX /opt/torch /opt/hf

# The weights are in the image, so the transformers loaders must not ask the hub whether a newer copy exists: without
# this flag a load waits on the network before it falls back to the cache, and a laptop without a connection waits long.
ENV HF_HUB_OFFLINE=1

USER student
WORKDIR /workspaces
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
