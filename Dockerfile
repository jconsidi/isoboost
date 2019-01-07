FROM ubuntu:latest

# baseline setup

RUN apt-get update

RUN apt-get install -y \
  build-essential \
  python3 \
  python3-dev \
  python3-pip

RUN pip3 install --upgrade pip

# custom start

ENV HOME /jc/numerai-model
WORKDIR $HOME

ADD requirements.txt $HOME
RUN pip3 install -r requirements.txt

ADD isoboost/ $HOME/isoboost
RUN python3 -m compileall isoboost/*.py
ADD setup.py $HOME
RUN python3 setup.py install

ADD test/*.py test/*.sh test/
