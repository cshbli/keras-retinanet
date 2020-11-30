FROM tensorflow/tensorflow:1.12.0-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \    
    libsm6 \
    libxext6 \
    libxrender1

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

RUN python -m pip install --user -U pip
RUN pip install opencv-python==3.4.5.20
RUN pip install rawpy
RUN pip install keras==2.2.4

ENV TF_CPP_MIN_LOG_LEVEL 3
