from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

from sparse import SparseRetriever
from dense import DenseRetriever, DenseConfig
from document_chunker import chunk, DocumentChunkerStrategy

from utils import collect_file_paths, AnswerKey, TestForm
from prompt import prompt

from constants import RAW_DATA_ROOT, DATA_ROOT

"""
Source: https://github.com/huggingface/huggingface-llama-recipes
"""

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# [NOTE]: do not use Instruct. it sucks.
model_id = "meta-llama/Llama-3.1-8B"

chunking_strategy_config = {
    "chunking_strategy" : DocumentChunkerStrategy.BY_WORD,
    "window_length" : 300,
    "overlap_length" : 100,
}

output_config = {
    "print_info" : False
}

folder_paths = [
    f"{DATA_ROOT}/raw2/text",
]

file_paths = []
chunks = []

for folder_path in folder_paths:
    file_paths.extend(collect_file_paths(folder_path))

for file_path in file_paths:
    chunks.extend(chunk(file_path, chunking_strategy_config, output_config))

print(f"Found {len(chunks)} chunks.")
# AFTER (uses Qwen embeddings)
USE_DENSE = True
if USE_DENSE:
    retriever = DenseRetriever(
        DenseConfig(
            model_name="Linq-AI-Research/Linq-Embed-Mistral",  # << switch to Qwen embeddings
            normalize=True,                        # L2-normalize for cosine via FAISS IP
            batch_size=8,                          # tune for VRAM; 8–16 is typical on A100
            max_length=2048                        # your chunks are ~300 words, so 2048 is plenty
        )
    ).fit(chunks)
else:
    retriever = SparseRetriever()
    retriever.build(chunks)


k = 2
QUESTIONS_FILE_PATH = f"{DATA_ROOT}/to-annotate/annotations/collated_questions.txt"
ANSWERS_FILE_PATH = f"{DATA_ROOT}/to-annotate/annotations/collated_references.json"
MODEL_ANSWERS_FILE_PATH = f"{DATA_ROOT}/to-annotate/annotations/system_output.json"
questions = TestForm(QUESTIONS_FILE_PATH)
ground_truth_answers = AnswerKey(ANSWERS_FILE_PATH)
model_answers = AnswerKey(MODEL_ANSWERS_FILE_PATH, form_mode=True)
NUM_QUESTIONS = 50

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

for q_num in range(1, 50):
    print(f"Answering question num {q_num}")
    query = questions.get_question(q_num)
    scored_doc_ids = retriever.search(query, k = k)
    retrieved_chunks = []
    for doc_id, score in scored_doc_ids:
        retrieved_chunks.append(retriever.get_doc(doc_id))

    qa_prompt = prompt(query, retrieved_chunks, k)

    outputs = pipe(
        qa_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.4,
        return_full_text=False
    )

    model_answer = outputs[0]['generated_text']  

    model_answers.submit_answer(q_num, model_answer)

print(model_answers.answer_key)
