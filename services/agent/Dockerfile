FROM quay.io/cdis/amazonlinux-base:master
ARG ROOT_PATH_V2=/qag/v2
ARG ROOT_PATH_V3=/qag/v3

RUN yum -y update && \
    yum install -y python3.12 python3.12-pip git && \
    yum clean all

RUN curl -fsSL https://rpm.nodesource.com/setup_24.x | bash - && \
    yum install -y nodejs

RUN git clone https://github.com/StevenSong/ai-chat-ui.git /build_v2
RUN npm --prefix /build_v2 install && \
    npm --prefix /build_v2 run build -- --base="$ROOT_PATH_V2/"

# 🤮 TODO: don't do this...
RUN git clone https://github.com/StevenSong/ai-chat-ui.git /build_v3
RUN npm --prefix /build_v3 install && \
    npm --prefix /build_v3 run build -- --base="$ROOT_PATH_V3/"

COPY services/agent/requirements.txt /requirements.txt
RUN pip3.12 install -r /requirements.txt

COPY services/agent/ /agent/
RUN pip3.12 install -e /agent/.
COPY model-config.yaml /agent/src/model-config.yaml
COPY model-config.local-vllm.yaml /agent/src/model-config.local-vllm.yaml
RUN cp -r /build_v2/dist/assets /agent/src/static/assets_v2 && \
    ln -sf /agent/src/static/assets_v2/index-*.js /agent/src/static/assets_v2/index.js && \
    ln -sf /agent/src/static/assets_v2/index-*.css /agent/src/static/assets_v2/index.css
RUN cp -r /build_v3/dist/assets /agent/src/static/assets_v3 && \
    ln -sf /agent/src/static/assets_v3/index-*.js /agent/src/static/assets_v3/index.js && \
    ln -sf /agent/src/static/assets_v3/index-*.css /agent/src/static/assets_v3/index.css

WORKDIR /agent/src
ENTRYPOINT ["python3.12", "server.py"]