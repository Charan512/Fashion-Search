# Fashion Retrieval System

> **Multi-vector fashion image search using CLIP + FashionCLIP + zero-shot attribute decomposition**

An intelligent multimodal search engine that retrieves fashion images based on natural language descriptions. Unlike vanilla CLIP, this system understands **compositionality**, **fine-grained fashion attributes**, and **contextual awareness**.

---

## 🚀 Quick Start

```bash
# 1. Clone and enter
git clone <repo-url>
cd Fashion-Search

# 2. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install git+https://github.com/openai/CLIP.git   # CLIP from source

# 3. Set secrets
cp .env.example .env
# → Fill in PINECONE_API_KEY and PINECONE_INDEX_NAME

# 4. Install spaCy language model
python -m spacy download en_core_web_sm

# 5. Build the index (1000 images for dev)
./scripts/build_index.sh

# 6. Launch the demo
streamlit run demo/app.py
```

---

## 🏗️ Architecture

```
USER QUERY (natural language)
        │
        ▼
┌──────────────────────┐
│   QueryDecomposer    │  ← dict-based, no ML required
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────┐
│  CLIP   │  │  FashionCLIP │
│  (50%)  │  │    (30%)     │
└────┬────┘  └──────┬───────┘
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │    Pinecone   │  ← 512-D cosine search
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │  Attr Matcher │  ← metadata scoring (20%)
    └───────┬───────┘
            ▼
    ┌───────────────┐
    │ ResultRanker  │  ← hard constraints + diversity
    └───────────────┘
```

### Scoring Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| CLIP (ViT-B/32) | **50%** | Global scene/context |
| FashionCLIP | **30%** | Fashion-domain fine-tuning |
| Attribute match | **20%** | Color, clothing, setting, formality |

---

## 📦 Project Structure

```
Fashion-Search/
├── config.yaml                   # Centralized configuration
├── .env.example                  # Secrets template (copy → .env)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
│
├── part_a_indexer/               # Offline indexing pipeline
│   ├── index.py                  # Main orchestrator (CLI)
│   ├── dataset_processor.py      # Image loading + Fashionpedia loader
│   ├── embedding_extractor.py    # CLIP + FashionCLIP embedding
│   ├── attribute_extractor.py    # Zero-shot attribute extraction
│   ├── vector_storage.py         # Pinecone interface
│   ├── utils/
│   │   ├── config_utils.py       # YAML + env loading
│   │   ├── image_utils.py        # PIL helpers
│   │   └── vector_utils.py       # L2, cosine, score combination
│   └── tests/
│
├── part_b_retriever/             # Online retrieval pipeline
│   ├── retriever.py              # FashionRetriever orchestrator
│   ├── query_processor.py        # QueryDecomposer
│   ├── multi_vector_search.py    # Parallel CLIP + FashionCLIP search
│   ├── attribute_matching.py     # AttributeMatcher
│   ├── ranker.py                 # ResultRanker (diversity + hard filters)
│   ├── utils/
│   │   ├── dictionaries.py       # Color/clothing/setting vocabularies
│   │   ├── explainability.py     # SearchResult + ExplainabilityEngine
│   │   └── prompt_utils.py       # CLIP prompt builders
│   └── tests/
│
├── demo/                         # Streamlit UI
│   ├── app.py                    # Entry point
│   ├── pages/
│   │   ├── 01_🔍_Search.py       # Main search interface
│   │   ├── 02_📚_Examples.py     # Evaluation queries showcase
│   │   └── 03_📊_About.py        # Architecture overview
│   └── components/
│       ├── theme.py              # Green/gold/black CSS injection
│       ├── result_card.py        # Result display card
│       ├── search_box.py         # Query input component
│       └── utils.py              # UI helpers
│
└── scripts/
    ├── build_index.sh            # Indexing shell wrapper
    └── evaluate.py               # Run 5 evaluation queries
```

---

## 🔧 Configuration

All configuration lives in [`config.yaml`](./config.yaml). Override individual values with environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PINECONE_API_KEY` | — | **Required** — your Pinecone API key |
| `PINECONE_INDEX_NAME` | `fashion-retrieval` | Index name to create/use |
| `DATASET_SUBSET_SIZE` | `1000` | Images to index (50000 for production) |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🔄 Part A — Indexer

```bash
# Dev subset (1000 images)
python -m part_a_indexer.index --subset 1000

# Production (50K images, resumable)
python -m part_a_indexer.index --subset 50000 --resume

# Dry run (extracts embeddings, no Pinecone writes)
python -m part_a_indexer.index --subset 100 --dry-run
```

The indexer stores for each image:
- **Primary vector**: CLIP global embedding (512-D)
- **Metadata**: FashionCLIP embedding, scene embedding, colors, clothing items, formality score, setting, style category

---

## 🔍 Part B — Retriever

```python
from part_b_retriever.retriever import FashionRetriever

retriever = FashionRetriever()
results = retriever.search("A red tie and white shirt in a formal office", top_k=10)

for r in results:
    print(f"#{results.index(r)+1} {r.image_id}: {r.overall_score:.3f}")
    print(f"  {r.explanation}")
```

---

## 🧪 Testing

```bash
# All tests (no GPU required — uses mocks)
pytest -v --tb=short

# Part A tests only
pytest part_a_indexer/tests/ -v

# Part B tests only  
pytest part_b_retriever/tests/ -v

# Run evaluation queries
python scripts/evaluate.py --top-k 10
```

---

## 🎨 Demo UI

```bash
streamlit run demo/app.py
```

Features:
- **Search page** — natural language query with results in 2-column grid
- **Examples page** — all 5 evaluation queries with descriptions
- **About page** — architecture diagram and scoring explanation

---

## ❓ Why Not Just CLIP?

Vanilla CLIP fails at:
- ❌ `"red tie + white shirt"` vs `"white tie + red shirt"` — treats as same vector
- ❌ Fashion-specific nuances (blazer collar style, formality)  
- ❌ Contextual awareness (office vs park)

Our system solves these with:
- ✅ `QueryDecomposer` — binds colors to specific items
- ✅ `AttributeMatcher` — explicit metadata filtering  
- ✅ FashionCLIP — fashion-domain fine-tuned embedding
- ✅ Hard constraint filtering — enforces formality requirements