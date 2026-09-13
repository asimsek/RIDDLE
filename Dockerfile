FROM gitlab-registry.nrp-nautilus.io/nrp/scientific-images/python:cuda-latest

USER root

LABEL org.opencontainers.image.source="https://github.com/asimsek/RIDDLE"

COPY requirements.txt /opt/riddle-image/requirements.txt

RUN /opt/conda/bin/python -m pip install --no-cache-dir \
        -r /opt/riddle-image/requirements.txt \
    && /opt/conda/bin/python -m pip check \
    && /opt/conda/bin/python -m pip freeze \
        > /opt/riddle-image/packages.lock

RUN /opt/conda/bin/python -c \
    "import h5py, matplotlib, mplhep, numpy, pandas, yaml, sklearn, scipy, tables, torch, tqdm, vector; from nflows.flows.base import Flow; assert torch.version.cuda is not None"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 1000:100
