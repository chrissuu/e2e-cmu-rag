from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from sparse import SparseRetriever
from document_chunker import chunk, DocumentChunkerStrategy

from utils import collect_file_paths
from prompt import prompt

from constants import RAW_DATA_ROOT

"""
Source: https://github.com/huggingface/huggingface-llama-recipes
"""


print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


model_id = "meta-llama/Llama-3.1-8B-Instruct"  # you can also use 70B, 405B, etc.

chunking_strategy_config = {
    "chunking_strategy" : DocumentChunkerStrategy.BY_SENTENCE,
    "window_length" : 8,
    "overlap_length" : 2,
}

output_config = {
    "print_info" : True
}

chunks = []
file_paths = collect_file_paths(f"{RAW_DATA_ROOT}/wikipedia/cmu-one-jump")
for file_path in file_paths:
    chunks.extend(chunk(file_path, chunking_strategy_config, output_config))

retriever = SparseRetriever()
retriever.build(chunks)

k = 5
query = "On what day did William S. Dietrich II pass away?"
scored_doc_ids = retriever.search(query, k = k)
retrieved_chunks = []
for doc_id, score in scored_doc_ids:
    retrieved_chunks.append(retriever.get_doc(doc_id))

qa_prompt = prompt(query, retrieved_chunks, k)

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
outputs = pipe(
    qa_prompt,
    max_new_tokens=20000,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(outputs[0]["generated_text"])
