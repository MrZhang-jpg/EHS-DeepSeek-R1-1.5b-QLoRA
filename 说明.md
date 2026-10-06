# DeepSeek‑R1‑1.5B EHS‑QLoRA 微调项目
> 基于 DeepSeek‑R1‑Distill‑Qwen‑1.5B 使用 QLoRA 做工地 EHS（安全生产）领域微调，训练数据包含法律法规、安全交底、事故案例、操作规程，微调后模型会输出 R1 风格``推理思考链再给出EHS专业答案。

## 📋 项目简介
本项目对 `deepseek‑ai/DeepSeek‑R1‑Distill‑Qwen‑1.5B` 执行4‑bit QLoRA微调，使用600条EHS领域带思考链的训练样本。
- 训练输出：LoRA适配器，可合并为完整HF模型，支持导出GGUF用于Ollama本地部署
- 特点：保留R1思考链输出，专门面向施工现场安全问答；适合做工地安全员助手、EHS知识库问答。

> ⚠️ 注意：本项目输出模型仅作为辅助参考，不能替代官方法规文件、持证安全人员的专业判断。

## 📂 项目目录结构
```
EHS‑R1‑1.5B‑QLoRA/
├── train_deepseek_r1_1.5b_qlora_sft.py      # QLoRA训练主脚本
├── nfer_deepseek_r1_1.5b_qlora.py          # LoRA适配器推理脚本
├── merge_lora_deepseek_r1_1.5b.py          # LoRA合并完整HF模型脚本
├── Dockerfile                               # 容器部署配置
├── Modelfile                                # Ollama导入配置模板
├── train.jsonl                              # 600条EHS训练数据集（R1‑think格式）
├── tools/                                   # 辅助工具脚本
│   └── *.py
├── .gitignore                               # git忽略配置
└── README.md                                # 本说明文档
```

> 📌 **大文件不存入Git/GitHub**
> `qlora‑model/`（合并完整模型）、`qlora‑mode/`（GGUF文件）、LoRA适配器、日志、checkpoint、虚拟环境、huggingface缓存全部通过`.gitignore`忽略，不提交至GitHub。

## 🛠️ 环境依赖
Python >=3.12
```bash
pip install torch transformers peft trl bitsandbytes datasets sentencepiece accelerate
```

## 📝 训练数据集格式
`train.jsonl`，每行一条R1蒸馏格式样本，必须包含``思考标签：
```json
{"messages": [
  {"role":"user","content":"工地动火、临时用电作业时，法律要求怎么管？"},
  {"role":"assistant","content":"问题拆解：考危险作业现场安全管理；依据定位：《安全生产法》第43条；判断：爆破、吊装、动火、临时用电属法定危险作业，必须安排专门人员进行现场安全管理。\n答案：生产经营单位进行爆破、吊装、动火、临时用电以及国务院应急管理部门会同国务院有关部门规定的其他危险作业，应当安排专门人员进行现场安全管理，确保操作规程的遵守和安全措施的落实。"}
]}
```

## 🚀 训练运行
### vGPU‑32G（支持bf16）推荐参数
- `lr=1e‑4`，`num_train_epochs=2`，`r=8`
```bash
python train_deepseek_r1_1.5b_qlora_sft.py
```
训练结束输出LoRA适配器到输出目录。

> 硬件提示：RTX30系列显卡不支持bf16，脚本内需要设置`fp16=True, bf16=False`。

## 🔍 推理方式
### 方式1：加载LoRA适配器推理
```bash
python nfer_deepseek_r1_1.5b_qlora.py
```

### 方式2：合并LoRA得到完整HF模型
```bash
python merge_lora_deepseek_r1_1.5b.py
```

### 方式3：导出GGUF并导入Ollama
1. 使用`llama.cpp`工具将合并后的HF模型转换GGUF（推荐量化等级`Q4_K_M`）
2. 基于项目内`Modelfile`执行ollama导入：
```bash
ollama create ehs‑r1:1.5b -f Modelfile
ollama run ehs‑r1:1.5b
```
> 💡提示：Ollama调用R1系列模型优先使用`/api/chat`接口，避免`generate`接口模板渲染异常。

## 🐳 Docker容器部署
构建镜像：
```bash
docker build -t ehs‑r1‑1.5b:latest .
```
> 说明：Dockerfile仅封装运行环境；模型权重（HF/GGUF）需要用户自行挂载到容器，不打包进镜像。

## ⚙️ 关键调参经验（针对本600条EHS数据集）
|配置项|推荐值|说明|
|---|---|---|
|LoRA‑r|8|不要过大，防止过拟合|
|learning_rate|1e‑4|1.5B小模型，不建议大于2e‑4|
|epoch|2|3轮极易发生输出退化、循环重复|
|batch_size|4|根据显存调整，配合梯度累积|

> 现象说明：小模型小数据集很容易出现**过拟合**，表现为训练loss很低，但实际推理出现循环文本、空think、无意义输出，优先降低epoch、学习率。