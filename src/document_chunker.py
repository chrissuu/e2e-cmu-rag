from enum import Enum

import re

from utils import Pipe
from constants import RAW_DATA_ROOT

class DocumentChunkerStrategy(Enum):
    BY_CHAR = 1
    BY_WORD = 2
    BY_SENTENCE = 3

def clean_text(text):
    """
    Cleans a text file by:
      - Removing extraneous whitespace
      - Ensuring exactly one space after common punctuation marks
      - Preserving regular spaces
    Saves the cleaned text to output_path.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', text)

    return text

def strip(text: str):
    return text.strip()

def split(text: str):
    return text.split(' ')

def split_sentences(text: str):
    return re.split(r'(?<=[.!?;])\s+', text.strip())

def split_container_with_overlap(container, window_length, overlap_length):
    """
    Chunks a container by window_length and overlap_length 
    padding on the left and right of the window.

    Note that this definition allows us to split by any container,
    such as by string, by sentence, by char, by word, etc.
    """
    chunks = []

    l_ptr = 0
    r_ptr = window_length
    while r_ptr + window_length < len(container):
        l_ptr_extended = l_ptr - overlap_length
        r_ptr_extended = r_ptr + overlap_length

        tmp_l_ptr = l_ptr
        tmp_r_ptr = r_ptr

        if l_ptr_extended >= 0:
            tmp_l_ptr = l_ptr_extended

        if r_ptr_extended < len(container):
            tmp_r_ptr = r_ptr_extended
                
        chunks.append(container[tmp_l_ptr:tmp_r_ptr])
        l_ptr = r_ptr
        r_ptr += window_length

    if r_ptr < len(container):
        chunks.append(container[r_ptr:])

    chunks = list(filter(lambda chunk : len(chunk) != 0, chunks))

    for chunk in chunks: 
        assert(len(chunk) <= window_length + 2 * overlap_length)

    return chunks


def chunk(file_path, chunking_strategy_info, output_info):
    chunking_strategy = chunking_strategy_info["chunking_strategy"]
    window_length = chunking_strategy_info["window_length"]
    overlap_length = chunking_strategy_info["overlap_length"]

    with open(file_path, "r", encoding="utf-8") as f:
        if chunking_strategy == DocumentChunkerStrategy.BY_CHAR:
            container = Pipe(f) | \
                    (lambda f : f.read()) | \
                    clean_text | \
                    strip

        elif chunking_strategy == DocumentChunkerStrategy.BY_WORD:
            container = Pipe(f) | \
                    (lambda f : f.read()) | \
                    clean_text | \
                    strip | \
                    split 

        elif chunking_strategy == DocumentChunkerStrategy.BY_SENTENCE:
            container = Pipe(f) | \
                    (lambda f : f.read()) | \
                    clean_text | \
                    strip | \
                    split_sentences

    split_chunks = split_container_with_overlap(container.value, window_length, overlap_length)

    # each chunk has been split, so we must join it back together
    joint_chunks = []
    for chunk in split_chunks:
        joint_chunks.append(' '.join(chunk))


    output_folder_path = output_info.get("folder_path", None)
    print_info = output_info.get("print_info", None)

    if output_folder_path:
        for i, chunk in enumerate(joint_chunks):
            f = open(f"{output_folder_path}/chunk_{i}_{window_length}_{overlap_length}.txt", "w")
            f.write(chunk)

    if print_info:
        print(f"""
        *=====================================================================================*
        |Processing {'/'.join(file_path.split('/')[5:])} for chunking.                  
        |Using {chunking_strategy.name} for the chunking strategy.               
        |Using window length {window_length} and overlap length {overlap_length}.
        |Found {len(joint_chunks)} chunks under this strategy.                   
        *=====================================================================================*
        """)
    return joint_chunks

chunking_strategy_info = {
    "chunking_strategy" : DocumentChunkerStrategy.BY_SENTENCE,
    "window_length" : 5,
    "overlap_length" : 1
}

output_info = {
    "print_info" : True
}

chunk(f"{RAW_DATA_ROOT}/wikipedia/cmu-one-jump/Carnegie_Mellon_University.txt", chunking_strategy_info, output_info)