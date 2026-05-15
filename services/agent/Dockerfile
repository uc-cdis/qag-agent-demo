FROM quay.io/cdis/amazonlinux-base:master

RUN yum -y update && \
    yum install -y python3.12 python3.12-pip && \
    yum clean all

COPY services/agent/requirements.txt /requirements.txt
RUN pip3.12 install -r /requirements.txt

COPY services/agent/ /agent/
COPY model-config.yaml /agent/src/model-config.yaml
WORKDIR /agent/src
ENTRYPOINT ["python3.12", "server.py"]
