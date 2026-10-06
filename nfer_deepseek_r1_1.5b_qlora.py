import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_path = "./deepseek_r1_1.5b_qlora_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
peft_model = PeftModel.from_pretrained(base_model, lora_path)

# DeepSeek R1 对话模板
prompt = "请简单解释什么是QLoRA"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = peft_model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
