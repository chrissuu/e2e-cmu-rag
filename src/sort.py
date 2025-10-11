import json
from constants import *
import json

def sort_questions_and_answers(questions_file: str, answers_file: str, output_questions: str, output_answers: str):
    """
    Sorts questions alphabetically and reorders answers so they match the new order.
    Assumes answers.json uses 1-based indexing for keys.
    """
    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    # Load answers
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    # Pair each question (1-indexed) with its answer
    question_answer_pairs = []
    for k, v in answers.items():
        try:
            idx = int(k) - 1  # convert to 0-based
            if 0 <= idx < len(questions):
                question_answer_pairs.append((questions[idx], v))
            else:
                print(f"⚠️ Skipping answer key {k}: index {idx} out of range.")
        except ValueError:
            print(f"⚠️ Skipping invalid key: {k}")

    if not question_answer_pairs:
        raise ValueError("No valid question–answer pairs found.")

    # Sort alphabetically by question
    question_answer_pairs.sort(key=lambda x: x[0].lower())

    # Rebuild outputs
    sorted_questions = [q for q, _ in question_answer_pairs]
    sorted_answers = {str(i + 1): a for i, (_, a) in enumerate(question_answer_pairs)}  # back to 1-indexed

    # Write sorted results
    with open(output_questions, 'w', encoding='utf-8') as f:
        f.write("\n".join(sorted_questions) + "\n")

    with open(output_answers, 'w', encoding='utf-8') as f:
        json.dump(sorted_answers, f, indent=2, ensure_ascii=False)

    print(f"✅ Sorted {len(sorted_questions)} questions to {output_questions}")
    print(f"✅ Wrote reordered answers to {output_answers}")


TEST_ROOT = f"{REPO_ROOT_PATH}/chrissu/data/test"
Q = f"{TEST_ROOT}/questions.txt"
sort_questions_and_answers(Q, f"{TEST_ROOT}/system_output_baseline.json", f"{TEST_ROOT}/sorted_questions.txt", f"{TEST_ROOT}/sorted_system_output_baseline.json")
sort_questions_and_answers(Q, f"{TEST_ROOT}/system_output_hybrid.json", f"{TEST_ROOT}/sorted_questions.txt", f"{TEST_ROOT}/sorted_system_output_hybrid.json")
sort_questions_and_answers(Q, f"{TEST_ROOT}/system_output_sparse.json", f"{TEST_ROOT}/sorted_questions.txt", f"{TEST_ROOT}/sorted_system_output_sparse.json")
sort_questions_and_answers(Q, f"{TEST_ROOT}/system_output_dense.json", f"{TEST_ROOT}/sorted_questions.txt", f"{TEST_ROOT}/sorted_system_output_dense.json")
