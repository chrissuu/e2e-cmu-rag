import os

class Pipe:
    def __init__(self, value):
        self.value = value

    def __or__(self, f):
        return Pipe(f(self.value))
    
    @classmethod
    def get_value(cls, pipe):
        return pipe.value

def collect_file_paths(folder_path):
    """
    Returns a list of all file paths under folder_path.
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            file_paths.append(os.path.join(root, f))
    return file_paths

def flatten(container_container):
    flattened_container = []
    for L in container_container:
        for e in L:
            flattened_container.append(e)

    return flattened_container

def make_human_readable(chunk: str):
    MAX_CHARS_PER_LINE = 80
    lines = []
    curr_line = []
    for word in chunk.split(' '):
        if len(' '.join(curr_line)) > MAX_CHARS_PER_LINE:
            lines.append(curr_line)
            curr_line = []
        curr_line.append(word)

    lines = list(map(lambda line_array : ' '.join(line_array), lines))
    return '\n'.join(lines)