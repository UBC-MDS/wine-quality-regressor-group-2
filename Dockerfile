FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

USER root

# install lmodern for Quarto PDF rendering
RUN sudo apt update \
    && sudo apt install -y lmodern

USER $NB_UID

RUN mamba update --quiet --file /tmp/conda-linux-64.lock && \
    mamba clean --all -y -f && \
    pip install altair-ally==0.1.1 deepchecks==0.18.1 && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"