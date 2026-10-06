# EHS-QLoRA DeepSeek-R1-1.5B Ollama 推理服务镜像
#
# 构建：docker build -t ehs-r1-ollama .
# 运行：docker run -d -p 11434:11434 --name ehs-r1 ehs-r1-ollama
# 调用：curl http://localhost:11434/api/generate -d '{"model":"ehs-r1:1.5b","prompt":"工地动火作业前需要办理什么手续？","stream":false}'

FROM ollama/ollama:latest

# 拷贝量化后的 GGUF 模型与导入配置
COPY qlora-mode/qlora-model-Q4_K_M.gguf /models/qlora-model-Q4_K_M.gguf
COPY Modelfile.ehs /Modelfile.ehs
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 监听所有网卡，供宿主机/局域网调用
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_KEEP_ALIVE=24h
EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
