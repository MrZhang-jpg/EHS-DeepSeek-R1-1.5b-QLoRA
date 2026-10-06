# 导入torch，pytorch基础库，张量、cuda、设备相关操作
import torch
# datasets库：huggingface数据集加载工具，用来读取jsonl训练集
from datasets import load_dataset
# transformers：huggingface核心库，模型、分词器、训练参数
from transformers import (
    AutoModelForCausalLM, # 自动加载因果语言模型（大模型基座）
    AutoTokenizer,        # 自动加载分词器
    BitsAndBytesConfig,   # 量化配置类，用于4bit量化QLoRA
    TrainingArguments,    # 训练超参配置
)
# peft：轻量微调库，专门做LoRA，只训练少量参数，不改动基座权重
from peft import LoraConfig, get_peft_model
# trl：transformers强化学习库，SFTTrainer专门用来做有监督微调
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ===================== 配置 =====================
# 基座模型名称，HF仓库地址 DeepSeek-R1-Distill-Qwen-1.5B
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 训练数据集路径，jsonl格式
dataset_path = "./train.jsonl"
# 训练完成后LoRA适配器保存文件夹
output_dir = "./deepseek_r1_1.5b_qlora_adapter"

# 4bit QLoRA量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 开启4bit量化加载模型，大幅降低显存占用
    bnb_4bit_quant_type="nf4",        # 量化类型：nf4，专门针对大模型优化的4bit量化格式
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用bf16，提升精度（GPU需要支持bf16）
    bnb_4bit_use_double_quant=True,   # 双重量化，进一步压缩显存
)

# 加载数据集：读取json文件，split="train"取训练集
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 加载分词器，trust_remote_code允许加载模型仓库里自定义代码
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen系列没有pad_token，用eos_token充当padding token
tokenizer.padding_side = "right"          # padding在句子右侧，因果LM推荐right padding

# ===================== 数据格式化函数（关键修复） =====================
# ① trl 0.9.x 的 SFTTrainer 不支持 dataset_text_field="messages"，必须用 formatting_func
# ② trl 0.9.x 以 batched=True 调用 formatting_func：入参是 batch 字典，
#    example["messages"] 是"多条样本的列表"，函数必须返回"文本列表"
# ③ DeepSeek-R1 蒸馏版的官方 chat template 会把 assistant 内容里的 <think>...</think>
#    自动剥掉（只留</think>之后的部分），所以不能用 apply_chat_template，必须手动
#    构造保留 think 推理链的 R1 文本格式
# ④ 用 DataCollatorForCompletionOnlyLM 屏蔽 user 部分 loss，只训练 assistant 回答
def _build_r1_text(messages):
    user = messages[0]["content"]
    assistant = messages[1]["content"]
    # R1 蒸馏格式：BOS + User 标记 + 问题 + Assistant 标记 + 完整回答(含think链) + EOS
    return (
        "<｜begin▁of▁sentence｜>"
        f"<｜User｜>{user}"
        f"<｜Assistant｜>{assistant}"
        "<｜end▁of▁sentence｜>"
    )


def formatting_func(batch):
    return [_build_r1_text(m) for m in batch["messages"]]

# 只对 assistant 部分计算 loss（DeepSeek R1 蒸馏格式的 assistant 标记）
response_template = "<｜Assistant｜>"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer
)

# 加载4bit量化基座模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # 传入4bit量化配置
    device_map="auto",              # 自动分配模型层到GPU/CPU，不用手动指定device
    trust_remote_code=True,
)
model.config.use_cache = False # 训练时关闭KV Cache，节省显存；推理的时候要打开
model.config.pretraining_tp = 1 # 张量并行度设为1，单卡训练固定写1

# ===================== LoRA 配置（适配DeepSeek R1 1.5B） =====================
# 基座本质是Qwen1.5架构，target_modules和Qwen一致
lora_config = LoraConfig(
    r=8,                              # LoRA秩，越大可学习参数越多，拟合能力越强，显存略增，常用8/16
    lora_alpha=16,                    # LoRA缩放系数，一般设为 r*2
    target_modules=[                  # 需要挂LoRA的模型层；这里只给q_proj v_proj加LoRA
        "q_proj", "v_proj",
        # 可选，增加拟合能力，显存占用略升："k_proj", "o_proj"
    ],
    lora_dropout=0.05,                # LoRA层dropout，防止过拟合
    bias="none",                      # 不训练bias参数
    task_type="CAUSAL_LM"             # 任务类型：因果语言模型（文本生成）
)
# 将LoRA挂载到量化后的模型上
model = get_peft_model(model, lora_config)
# 打印可训练参数占总参数比例，QLoRA下一般只有0.1%左右参数参与训练
model.print_trainable_parameters()

# ===================== 训练参数 =====================
training_args = TrainingArguments(
    output_dir=output_dir,            # 模型保存目录
    per_device_train_batch_size=4,    # 单卡batch size，每个step每张卡一次读4条样本
    gradient_accumulation_steps=2,    # 梯度累积：累计2个step再更新权重，等效总batch=4*2=8
    learning_rate=1e-4,               # 学习率：经验证1e-4更稳（2e-4+3epoch对1.5B易过拟合退化）
    num_train_epochs=2,               # 训练轮数：600条数据2轮足够（3轮已验证过拟合，输出退化）
    logging_steps=5,                  # 每5个step打印一次loss日志
    save_strategy="epoch",            # 保存策略：每跑完1个epoch保存一次LoRA权重
    # 自动判断GPU是否支持bf16，支持就用bf16，不支持就用fp16
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit",         # 8bit优化器，QLoRA标配，减少优化器显存开销
    report_to="none",                 # 关闭wandb等在线日志工具，不做可视化上报
    save_total_limit=2,               # 最多保留2个checkpoint，旧的自动删除，防止占磁盘
)

# SFTTrainer：专门用于有监督微调的训练器（trl库）
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,      # 关键修复：用格式化函数把messages转成字符串（trl 0.9.x不支持dataset_text_field="messages"）
    max_seq_length=1024,                  # 样本最大长度，超过会截断
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,          # 只对assistant部分（含think链）计算loss
)

# 启动训练循环
trainer.train()
# 训练结束保存LoRA适配器（不是完整基座模型，只是几十MB小权重）
trainer.save_model(output_dir)
print(f"LoRA适配器保存至: {output_dir}")
