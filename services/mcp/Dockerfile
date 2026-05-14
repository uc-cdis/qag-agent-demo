FROM quay.io/cdis/amazonlinux-base:master

RUN yum -y update && \
    yum install -y python3.12 python3.12-pip && \
    yum clean all

COPY services/mcp/requirements.txt /requirements.txt
RUN pip3.12 install -r requirements.txt

COPY services/mcp/ /mcp/
WORKDIR /mcp/src
ENTRYPOINT ["python3.12", "server.py"]
