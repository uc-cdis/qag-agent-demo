FROM vllm/vllm-openai:latest

# need to patch for FIPS, see: https://github.com/vllm-project/vllm/issues/33147
RUN uv pip install --system opencv-python-headless==4.12.0.88
