from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

"""
Source: https://github.com/huggingface/huggingface-llama-recipes
"""


print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


model_id = "meta-llama/Llama-3.1-8B-Instruct"  # you can also use 70B, 405B, etc.

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model (optionally with 8-bit quantization to save VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",       # fp16/bf16 if supported
    load_in_8bit=True        # set True if using bitsandbytes
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Run inference
prompt = "Explain reinforcement learning in simple terms."
outputs = pipe(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(outputs[0]["generated_text"])
