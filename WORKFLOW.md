# Fashion Retrieval System: Project Workflow

This document illustrates the end-to-end data flow of the system. The architecture is strictly divided into two distinct pipelines: **Offline Indexing** (processing and storing the data) and **Online Retrieval** (handling user search queries in real-time).

## Complete Workflow Diagram

```mermaid
flowchart TD
    %% Define Styles
    classDef offline fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef online fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000

    %% ---------------------------------------------
    %% OFFLINE INDEXING PIPELINE
    %% ---------------------------------------------
    subgraph Offline["Offline Indexing Pipeline (part_a_indexer)"]
        direction TB
        
        DS[(Ashraq Dataset <br/> 44k Images)]:::storage
        
        IP[Dataset Processor <br/> Image Preprocessing]:::offline
        
        subgraph Models_Off["AI Feature Extraction"]
            EE[Embedding Extractor <br/> CLIP + FashionCLIP]:::model
            AE[Attribute Extractor <br/> Zero-Shot Prompting]:::model
        end
        
        ThumbScript[[build_thumbnails.py <br/> Base64 Image Compression]]:::offline

        DS --> IP
        DS --> ThumbScript
        
        IP --> EE
        IP --> AE
        
        EE -- Generates Vectors --> DB[(Pinecone Vector Database)]:::storage
        AE -- Generates Metadata --> DB
        
        ThumbScript -- Generates JSON Cache --> LocalStorage[(thumbnails.json)]:::storage
    end

    %% ---------------------------------------------
    %% ONLINE RETRIEVAL PIPELINE
    %% ---------------------------------------------
    subgraph Online["Online Retrieval Pipeline (part_b_retriever)"]
        direction TB
        
        User((User Search Query))
        
        QD[Query Decomposer <br/> SpaCy NLP Entity Extraction]:::online
        
        subgraph Models_On["Query Embedding"]
            Q_CLIP[CLIP Text Encoder]:::model
            Q_FCLIP[FashionCLIP Text Encoder]:::model
        end
        
        MVS[Multi-Vector Search <br/> Parallel Execution]:::online
        
        AM[Attribute Matcher <br/> Cross-checking metadata]:::online
        RR[Result Ranker <br/> Diversity + Hard Constraints]:::online
        UI[[Streamlit UI Dashboard]]:::online

        User --> QD
        
        QD -- Semantic Concepts --> Q_CLIP
        QD -- Fashion Entities --> Q_FCLIP
        QD -- Explicit Rules (e.g., Color) --> AM
        
        Q_CLIP --> MVS
        Q_FCLIP --> MVS
        
        MVS -- Cosine Similarity Query --> DB
        DB -- Returns Top K Candidates --> AM
        
        AM -- Adjusts Scores --> RR
        RR -- Final Sorted List --> UI
        
        LocalStorage -- Renders Images Offline --> UI
    end

    %% Connect offline to online dependencies
    DB -. Serves Searches .-> MVS
```

## Component Breakdown

### Part A: Offline Indexing
The offline process runs asynchronously to populate the database so that searches at runtime are nearly instantaneous.
1. **Dataset**: We pull ~44,000 images from the `ashraq/fashion-product-images-small` dataset.
2. **Thumbnails**: Because public image URLs break often, we generate a local base64 `thumbnails.json` cache. This ensures the Streamlit UI can render images 100% offline with zero latency.
3. **Embedding Extractor**: Each image is encoded into two distinct 512-dimension vectors—one by standard CLIP (for broad semantic understanding) and one by FashionCLIP (for domain-specific apparel understanding).
4. **Attribute Extractor (Zero-Shot)**: Instead of training classification models, we use CLIP to evaluate prompts (e.g., *"This is a formal outfit"*) against the image. The highest scoring prompts are attached to the Pinecone vector as searchable metadata.

### Part B: Online Retrieval
When a user types a query like *"red formal shirt for an office meeting"*:
1. **Query Decomposer**: SpaCy instantly parses the query to extract explicit constraints (Color = `red`, Item = `shirt`, Formality = `formal`, Setting = `office`).
2. **Parallel Search**: The query text is vectorized by both CLIP and FashionCLIP simultaneously. These vectors query their respective namespaces in Pinecone.
3. **Attribute Matcher & Ranker**: The system retrieves the top candidate images from Pinecone based purely on vector similarity. It then boosts the scores of items whose pre-computed metadata explicitly matches the constraints extracted by the Query Decomposer. Finally, hard constraints (e.g., dropping casual images for a formal search) are applied before sending the final ranked list to Streamlit.
