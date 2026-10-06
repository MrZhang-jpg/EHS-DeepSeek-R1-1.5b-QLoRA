import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_path = "./deepseek_r1_1.5b_qlora_adapter"
save_path = "./deepseek-r1-1.5b-merged"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
peft_model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("合并完成")
