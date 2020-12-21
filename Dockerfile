FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace
ENV MLFLOW_TRACKING_URI file:/app/workspace/mlruns

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# to support install openjdk-11-jre-headless
RUN mkdir -p /usr/share/man/man1
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    git \
    bzip2 \
    ca-certificates \
    libssl-dev \
    libmysqlclient-dev \
    default-libmysqlclient-dev \
    make \
    cmake \
    protobuf-compiler \
    curl \
    sudo \
    software-properties-common \
    xz-utils \
    file \
    mecab \
    libmecab-dev \
    python3-pip \
    openjdk-11-jre-headless \
    && curl -sL https://deb.nodesource.com/setup_10.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc
RUN pip3 install -U pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
#     pip install hydra-core --upgrade
COPY *.py ./
COPY *.sh ./
COPY workspace/data/* /app/workspace
# COPY config.yaml .
RUN mkdir -p $WORK_DIR/mlruns

CMD ["bash", "./run_ner.sh"]