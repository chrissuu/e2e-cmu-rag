import os
import json

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

import json

class TestForm:
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        # Strip newlines and build dict
        self.test_form = {str(i + 1): line.strip() for i, line in enumerate(lines)}

    def get_question(self, num):
        return self.test_form.get(str(num), None)


class AnswerKey:
    def __init__(self, path, form_mode=False):
        self.path = path
        self.form_mode = form_mode

        if form_mode:
            self.answer_key = {}
        else:
            with open(path, "r") as f:
                self.answer_key = json.load(f)

    def get_answer(self, key):
        return self.answer_key.get(str(key), None)
    
    def submit_answer(self, q_num, answer):
        self.answer_key[str(q_num)] = answer

        if not self.form_mode:
            with open(self.path, "w") as f:
                json.dump(self.answer_key, f, indent=2)
