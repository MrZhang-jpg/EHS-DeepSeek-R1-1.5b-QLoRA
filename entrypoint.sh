#!/bin/bash
# EHS-QLoRA Ollama 容器入口：启动 serve → 首次导入模型 → 保持前台运行
set -e

# 后台启动 ollama serve
ollama serve &
SERVE_PID=$!

# 等待 API 就绪
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

# 首次启动时导入 EHS 模型
if ! ollama list | grep -q "ehs-r1:1.5b"; then
  echo "[entrypoint] importing ehs-r1:1.5b ..."
  ollama create ehs-r1:1.5b -f /Modelfile.ehs
fi

echo "[entrypoint] ehs-r1:1.5b ready on 0.0.0.0:11434"
wait $SERVE_PID
