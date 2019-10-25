FROM nvcr.io/nvidia/pytorch:18.08-py3

USER root

################## DONT CHANGE BELOW ########################3
# Common DLI installs/config
# Install nginx version with extras
RUN apt-get update && apt-get install -y wget && wget -qO - https://openresty.org/package/pubkey.gpg | apt-key add - && apt-get -y install software-properties-common && add-apt-repository -y "deb http://openresty.org/package/ubuntu $(lsb_release -sc) main" && apt-get update && apt-get install -y --no-install-recommends openresty supervisor curl wget git && rm -rf /var/lib/apt/lists/*
#add-apt-repository -y ppa:nginx/stable && apt-get -y update &&  apt-get install -y  --no-install-recommends nginx supervisor curl wget git && rm -rf /var/lib/apt/lists/*

# RUN mkdir /dli
# WORKDIR /dli

# DIGITS env vars, not used everywhere, but keep them here as common globals anyways
# ENV DIGITS_URL_PREFIX=/digits
# ENV DIGITS_JOBS_DIR=/dli/data/digits
# ENV DIGITS_LOGFILE_FILENAME=$DIGITS_JOBS_DIR/digits.log

################## DONT CHANGE ABOVE ########################3

################## BASE SERVICES BELOW, CHANGE WITH CAUTION ########################3
# Install Jupyter, etc.
RUN pip install -U cython pip
RUN pip install --ignore-installed ipython jupyter
################## BASE SERVICES ABOVE, CHANGE WITH CAUTION ########################3


################## TASK SPECIFIC BELOW, CHANGE AS NEEDED ########################3

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \ 
      graphviz \
      make \
      cmake \
      wget \
      libz-dev \ 
      libxml2-dev \
      libopenblas-dev \
      libopencv-dev \
      graphviz-dev \
      libgraphviz-dev \
      ca-certificates \
      ffmpeg \
      unzip && \
    pip install -U \
      torch \
      numpy \
      scipy \
      networkx \
      matplotlib \
      sklearn \
      graphviz \
      nltk \
      requests[security]

# install mxnet nightly build
RUN pip install --pre mxnet-cu90

# install dgl (latest)
RUN pip install dgl==0.2

# remove pathlib package to use builtin pathlib
RUN pip uninstall pathlib -y

ENV PYTHONWARNINGS="ignore"
#RUN echo "import warnings; warnings.filterwarnings('ignore')" >> /root/.ipython/profile_default/startup/disable_warnings.py
#COPY dli/service/jupyter/custom/* /root/.jupyter/custom/
################## TASK SPECIFIC ABOVE, CHANGE AS NEEDED ########################3
