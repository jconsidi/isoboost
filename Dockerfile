FROM ubuntu:latest

# basic environment

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# baseline setup

RUN apt-get update

RUN apt-get install -y \
  build-essential \
  git \
  python3 \
  python3-dev \
  python3-pip

RUN pip3 install --upgrade pip

# custom start

ENV HOME /isoboost
WORKDIR $HOME

ADD requirements.txt $HOME
RUN pip3 install -r requirements.txt

ADD isoboost/ $HOME/isoboost
RUN python3 -m compileall isoboost/*.py
ADD setup.py $HOME
RUN python3 setup.py install

ADD examples/*.py examples/
ADD tests/*.py tests/*.sh tests/

RUN python3 -m compileall scripts/*.py
