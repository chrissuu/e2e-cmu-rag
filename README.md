For running on Google Colab (recommended)
# Core libs
!pip -q install sentence-transformers transformers accelerate bitsandbytes rank-bm25 \
               readability-lxml pdfplumber pypdf beautifulsoup4 lxml tqdm requests

# Try FAISS-GPU (works on many Colab images). If it fails, we fall back to faiss-cpu.
try:
    # Known good version on many CUDA12 Colab images
    !pip -q install faiss-gpu==1.7.4
    import faiss
    print("FAISS GPUs detected:", faiss.get_num_gpus())
except Exception as e:
    print("faiss-gpu import failed, falling back to faiss-cpu:", e)
    !pip -q install faiss-cpu
    import faiss
    print("FAISS (CPU) loaded OK")


THEN RUN:


%cd /content/e2e-cmu-rag/src
# Set env var your code requires
import os; os.environ["E2E_CMU_RAG"] = "/content/e2e-cmu-rag"
# Now run main
!python main.py


EVAL:

cd into e2e-cmu-rag/

run:

python src/eval.py \
  --gold data/to-annotate/annotations/collated_references.json \
  --pred data/to-annotate/annotations/system_output.json \
  --dump-per-item outputs/per_item_scores.json
