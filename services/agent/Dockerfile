FROM quay.io/cdis/amazonlinux-base:master
ARG ROOT_PATH=/qag

RUN yum -y update && \
    yum install -y python3.12 python3.12-pip git && \
    yum clean all

RUN curl -fsSL https://rpm.nodesource.com/setup_24.x | bash - && \
    yum install -y nodejs

COPY services/agent/requirements.txt /requirements.txt
RUN pip3.12 install -r /requirements.txt

COPY services/agent/ /agent/
COPY model-config.yaml /agent/src/model-config.yaml
COPY model-config.local-vllm.yaml /agent/src/model-config.local-vllm.yaml

RUN git clone https://github.com/StevenSong/ai-chat-ui.git /build
RUN npm --prefix /build install && \
    npm --prefix /build run build -- --base="$ROOT_PATH/"
RUN cp -r /build/dist/assets /agent/src/static/assets && \
    ln -sf /agent/src/static/assets/index-*.js /agent/src/static/assets/index.js && \
    ln -sf /agent/src/static/assets/index-*.css /agent/src/static/assets/index.css

WORKDIR /agent/src
ENTRYPOINT ["python3.12", "server.py"]