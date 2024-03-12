FROM tensorflow/tensorflow:2.11.0-gpu

ARG DEBIAN_FRONTEND=noninteractive
ARG GPUS=all

# Install apt dependencies
RUN apt-get update -y && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    python3-numpy \
    python3-opencv \
    python3-rasterio

RUN python -m pip install opencv-python

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy the tensorflow models code
RUN git clone https://github.com/tensorflow/models.git

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

# src: https://github.com/qubvel/segmentation_models/issues/374#issuecomment-672694688
ENV SM_FRAMEWORK tf.keras

WORKDIR /home/tensorflow/
# Install additional Python packages from requirements.txt
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
# Install any necessary dependencies
RUN pip install --no-cache-dir numpy rasterio image_slicer opencv-python-headless geopandas


