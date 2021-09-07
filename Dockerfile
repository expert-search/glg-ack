## Tensorflow
# Adapted from Tensorflow CPU
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu.Dockerfile
ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=2.6.0
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

## App
WORKDIR /app

COPY ./models models
COPY ./src/deployment src/deployment
COPY ./setup.py ./
COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app/src/deployment

CMD ["/bin/sh", "-c", "streamlit run app.py --browser.serverAddress 0.0.0.0 --server.port 80"]