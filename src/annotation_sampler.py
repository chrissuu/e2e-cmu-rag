from document_chunker import chunk, DocumentChunkerStrategy
from utils import collect_file_paths, flatten
from constants import ANNOTATION_DATA_ROOT, RAW_DATA_ROOT
import random
import os
import math
import shutil
import json

TEST_SIZE = 150
NUM_GROUPS = 3
SEED=19

assert(TEST_SIZE % NUM_GROUPS == 0)

random.seed(SEED)

chunking_strategy_config = {
    "chunking_strategy" : DocumentChunkerStrategy.BY_WORD,
    "window_length" : 1000,
    "overlap_length" : 100,
}

output_config = {
    "print_info" : False
}
folders_to_chunk = [RAW_DATA_ROOT]
file_paths_to_chunk = flatten(list(map(collect_file_paths, folders_to_chunk)))
chunks = list(map(lambda file_path: chunk(file_path, chunking_strategy_config, output_config), file_paths_to_chunk))
chunks = flatten(chunks)
print(f"Found {len(chunks)} different chunks.")
print(f"Sampling {TEST_SIZE} chunks from this population.")

assert len(chunks) >= TEST_SIZE

random.seed(SEED)
random_chunk_sample = random.sample(chunks, TEST_SIZE)

group_size = math.ceil(len(random_chunk_sample) / NUM_GROUPS)
output_root = ANNOTATION_DATA_ROOT
os.makedirs(output_root, exist_ok=True)

for i in range(NUM_GROUPS):
    group_dir = os.path.join(output_root, f"group_{i}")
    os.makedirs(group_dir, exist_ok=True)
    
    start_idx = i * group_size
    end_idx = min((i + 1) * group_size, len(random_chunk_sample))
    group_files = random_chunk_sample[start_idx:end_idx]
    
    for j, chunk_data in enumerate(group_files):
        chunk_name = f"chunk_{j}.txt"
        chunk_path = os.path.join(group_dir, chunk_name)
        
        if isinstance(chunk_data, str) and os.path.exists(chunk_data):
            shutil.copy(chunk_data, chunk_path)
        
        elif isinstance(chunk_data, str):
            with open(chunk_path, "w") as f:
                f.write(chunk_data)
        
        else:
            with open(chunk_path.replace(".txt", ".json"), "w") as f:
                json.dump(chunk_data, f, indent=2)
    
    info_path = os.path.join(group_dir, "__info__.txt")
    with open(info_path, "w") as f:
        f.write(f"Folders chunked: {folders_to_chunk}\n")
        f.write(f"Random seed used: {SEED}\n")
        f.write(f"Group index: {i}\n")
        f.write(f"Number of files: {len(group_files)}\n")
        f.write(f"Start index: {start_idx}, End index: {end_idx}\n")

print(f"Created {NUM_GROUPS} annotation folders in '{output_root}'")