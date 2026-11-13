ARG ROOT_PATH_V2=/qag/v2
ARG ROOT_PATH_V3=/qag/v3
ARG ROOT_PATH_SR=/qag/sr-chat

# from https://github.com/StevenSong/ai-chat-ui/actions
ARG UI_TAG=sha-dfc4aa3b002b477e24a787f4d63e2abfa501bef6

FROM ghcr.io/stevensong/ai-chat-ui-dist:${UI_TAG} AS ui

FROM quay.io/cdis/amazonlinux-base:master AS agent

RUN yum -y update && \
    yum install -y python3.12 python3.12-pip git findutils && \
    yum clean all

COPY --from=ui /dist /opt/ai-chat-ui

COPY services/agent/requirements.txt /requirements.txt
RUN pip3.12 install -r /requirements.txt

COPY services/agent/ /agent/
RUN pip3.12 install -e /agent/.
COPY model-config.yaml /agent/src/model-config.yaml
COPY model-config.local-vllm.yaml /agent/src/model-config.local-vllm.yaml

# inherit from outer scope
ARG ROOT_PATH_V2
ARG ROOT_PATH_V3
ARG ROOT_PATH_SR

RUN ls /agent

RUN bash /agent/install-ui.sh /opt/ai-chat-ui/assets /agent/src/static/assets_v2 "$ROOT_PATH_V2" && \
    bash /agent/install-ui.sh /opt/ai-chat-ui/assets /agent/src/static/assets_v3 "$ROOT_PATH_V3" && \
    bash /agent/install-ui.sh /opt/ai-chat-ui/assets /agent/src/static/assets_sr "$ROOT_PATH_SR"

WORKDIR /agent/src
ENTRYPOINT ["python3.12", "server.py"]