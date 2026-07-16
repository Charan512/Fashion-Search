# Fashion Retrieval System — Setup Guide

Step-by-step installation for macOS / Linux. Windows users should use WSL2.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | ≥ 3.8 | `python3 --version` |
| pip | any | `python3 -m pip --version` |
| Git | any | `git --version` |
| Pinecone account | free tier | [pinecone.io](https://www.pinecone.io/) |
| GPU (optional) | CUDA 11.8+ | `nvidia-smi` |

---

## Step 1 — Clone the Repository

```bash
git clone <repo-url>
cd Fashion-Search
```

---

## Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)
```

---

## Step 3 — Install Python Dependencies

```bash
# Install package in editable mode (sets up imports)
python3 -m pip install -e .

# Install CLIP from source (not on PyPI)
python3 -m pip install git+https://github.com/openai/CLIP.git

# Install all remaining dependencies
python3 -m pip install -r requirements.txt
```

> [!NOTE]
> This installs PyTorch, Transformers (FashionCLIP), Pinecone, Streamlit, and all other packages.
> Total download is ~2–4 GB depending on your CUDA version.

### GPU (CUDA) installation

If you have an NVIDIA GPU, install the CUDA-enabled PyTorch first:

```bash
# PyTorch with CUDA 11.8
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
python3 -m pip install -e .
python3 -m pip install git+https://github.com/openai/CLIP.git
```

---

## Step 4 — Set Up Pinecone

1. Sign up for a free account at [https://www.pinecone.io/](https://www.pinecone.io/)
2. Create a new Project
3. Copy your **API Key** from the API Keys page

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
PINECONE_API_KEY=your_actual_api_key_here
PINECONE_INDEX_NAME=fashion-retrieval    # can be any name
DATASET_SUBSET_SIZE=1000                 # 1000 for dev, 50000 for production
DEVICE=cpu                               # or cuda if GPU available
```

---

## Step 5 — Verify Installation

```bash
# Check CLIP loads correctly
python3 -c "import clip; model, _ = clip.load('ViT-B/32', device='cpu'); print('✓ CLIP OK')"

# Check FashionCLIP loads correctly
python3 -c "from transformers import CLIPModel; CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); print('✓ FashionCLIP OK')"

# Check Pinecone connects
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os; from pinecone import Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
print('✓ Pinecone connected. Indexes:', [i.name for i in pc.list_indexes()])
"

# Run the test suite (no GPU needed)
python3 -m pytest part_a_indexer/tests/ part_b_retriever/tests/ -v --tb=short \
  --ignore=part_a_indexer/tests/test_embedding_extractor.py \
  --ignore=part_a_indexer/tests/test_attribute_extractor.py
```

---

## Step 6 — Build the Index (Part A)

### Dev subset (1,000 images — recommended first run)

```bash
./scripts/build_index.sh
# OR
python3 -m part_a_indexer.index --subset 1000
```

### Test without Pinecone writes

```bash
python3 -m part_a_indexer.index --subset 100 --dry-run
```

### Full production index (50,000 images)

```bash
./scripts/build_index.sh --full
# OR
python3 -m part_a_indexer.index --subset 50000
```

### Resume an interrupted indexing run

```bash
./scripts/build_index.sh --resume
```

> [!NOTE]
> Indexing 1,000 images on CPU takes ~10–20 minutes.
> Indexing 50,000 images on GPU (A100) takes ~2–4 hours.

---

## Step 7 — Run the Streamlit Demo

```bash
streamlit run demo/app.py
```

Opens at [http://localhost:8501](http://localhost:8501) in your browser.

---

## Step 8 — Run Evaluation Queries

```bash
python3 scripts/evaluate.py --top-k 10

# Save results to JSON
python3 scripts/evaluate.py --top-k 10 --output-json results/evaluation.json
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'clip'`
```bash
python3 -m pip install git+https://github.com/openai/CLIP.git
```

### `ModuleNotFoundError: No module named 'part_a_indexer'`
```bash
python3 -m pip install -e .          # run from project root
```

### `EnvironmentError: Required environment variable 'PINECONE_API_KEY' is not set`
```bash
cp .env.example .env                 # copy template
# Edit .env and add your key
```

### `pinecone.exceptions.PineconeApiException: 401 Unauthorized`
- Double-check your `PINECONE_API_KEY` in `.env`
- Ensure the key is for the correct Pinecone project

### CUDA Out of Memory during indexing
```bash
# Reduce batch size
python3 -m part_a_indexer.index --subset 1000 --batch-size 8
```

### FashionCLIP download is slow / fails
FashionCLIP is ~1.7 GB. If the download times out, the system automatically falls back
to CLIP-only for fashion embeddings (30% weight becomes 0%, semantic gets 80%). Results
are still useful.

---

## Project Layout Quick Reference

```
Fashion-Search/
├── .env                      ← your secrets (git-ignored)
├── config.yaml               ← centralized config
├── requirements.txt          ← Python deps
│
├── part_a_indexer/           ← offline indexing
│   └── index.py              ← python3 -m part_a_indexer.index
│
├── part_b_retriever/         ← online retrieval
│   └── retriever.py          ← FashionRetriever class
│
├── demo/
│   └── app.py                ← streamlit run demo/app.py
│
└── scripts/
    ├── build_index.sh        ← indexing shortcut
    └── evaluate.py           ← 5 evaluation queries
```
