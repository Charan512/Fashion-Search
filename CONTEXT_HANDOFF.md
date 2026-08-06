# Context Handoff: Gradio Migration for Fashion Retrieval System

This document summarizes the current state of the repository and the planned next steps for migrating the frontend from Streamlit to Gradio.

---

## 📌 Project Overview
- **System**: Multi-vector fashion image search engine using CLIP + FashionCLIP + zero-shot attribute decomposition.
- **Backend/Database**: Pinecone serverless index (`fashion-retrieval`) populated with dual-namespace vectors.
- **Dataset**: `ashraq/fashion-product-images-small` (44,200 images).

---

## ⚡ Current Status
1. **Indexing Complete**: 100% of the 44,200 images are fully indexed and vectors are populated in Pinecone.
2. **Tests Verified**: The test suite is 100% healthy. All **110 unit tests are passing** (`pytest -v --tb=short`). Recent regression fixes:
   - Fixed PyTorch dimension expansion mismatch in the `mock_encode_text` helper within `test_attribute_extractor.py` (changed `scores.unsqueeze(0)` to `scores.unsqueeze(1)`).
   - Updated mock patches in `test_attribute_extractor.py` to patch `_score_prompts_fast` instead of the legacy `_score_prompts` method.
   - Patched `_load_fashion_clip` in `test_embedding_extractor.py` to properly test the FashionCLIP fallback behavior without triggering lazy loading.
3. **Frontend Status**: The app is currently using **Streamlit** (located in `demo/app.py` and the `demo/pages` directory).

---

## 🚀 Next Step: Migration to Gradio
The objective is to replace the Streamlit frontend with a **Gradio** web interface.

### Existing Streamlit App structure to replicate:
- **`demo/app.py`**: Entry point and custom theme.
- **`demo/pages/01_Search.py`**: Core Search page.
  - Multi-vector scoring model setup: Semantic weight (50%), Fashion weight (30%), Attribute weight (20%).
  - Text search field and option to query.
  - Interactive filters: style category, setting, formality, and color/clothing items constraints.
  - Result display card showing score breakdown (Semantic + Fashion + Attribute Match) and highlighted attributes.
- **`demo/pages/02_Examples.py`**: Evaluation query cases showing common fashion search terms and their behavior.
- **`demo/pages/03_About.py`**: Architecture diagram (Mermaid.js workflow) and system breakdown.

### Gradio Migration Plan:
1. Create a Gradio interface (e.g. `demo/gradio_app.py` or similar).
2. Wire the input fields (text search query, weight sliders, attribute filters) to the `FashionRetriever` orchestrator.
3. Construct the output components (e.g. a grid showing retrieved clothing items with matching attributes and score breakdown).
4. Verify the Gradio application works locally by running it.
