# Code Markdown using CodeLlama-7B-Instruct (GGUF) with RAG and Gradio

This repository demonstrates a local, self-contained Retrieval-Augmented Generation (RAG) pipeline for source code understanding and contextual querying. It leverages the quantized `CodeLlama-7B-Instruct` model in GGUF format, optimized to run on Apple Silicon (M1/M2/M3) using `llama-cpp-python`. Explanations and embeddings are stored in a persistent vector database (ChromaDB).

## Features
- Locally run quantized LLM (no GPU required)
- Supports `.py`, `.cs`, `.cpp`, etc. code files
- Generates detailed, step-by-step code explanations
- Stores context in ChromaDB for retrieval
- Enables question-answering with RAG-style prompting

## System Requirements
- macOS (tested on M1 Pro with 16GB RAM)
- Python 3.10+
- Minimum 8 GB RAM (16 GB recommended)
- No CUDA/GPU needed (CPU inference only)

## Architecture Overview
1. A quantized CodeLlama model is loaded using `llama-cpp-python`.
2. Code snippets are passed through prompt templates to extract detailed natural language explanations.
3. These explanations are embedded using `sentence-transformers` and stored in ChromaDB.
4. User questions are vectorized and matched against stored contexts.
5. Relevant context is passed back to the model for answer generation.

## Environment Setup

### Step 1: Install Anaconda
Download and install from:
https://www.anaconda.com/products/distribution#download-section

### Step 2: Create Conda Environment
```bash
conda create -n code_llama_env python=3.10 -y
conda activate code_llama_env
```

### Step 3: Clone the Repository
```bash
git clone https://github.com/yourusername/code-explainer-llama.git
cd code-explainer-llama
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
> If `requirements.txt` is missing, install manually:
```bash
pip install llama-cpp-python huggingface_hub chromadb sentence-transformers python-dotenv gradio
```

### Step 5: Hugging Face Access Token
1. Visit https://huggingface.co/settings/tokens and generate a Read access token
2. Create a `.env` file in the project root with:
```
HF_TOKEN=your_token_here
```

## Running the Project

### Option 1: Jupyter Notebook
1. Launch Jupyter:
```bash
jupyter notebook
```
2. Open the notebook file (e.g., `code_explainer_mac_only.ipynb`)
3. Follow each cell sequentially — model will be downloaded, code explained, vectors stored, and Gradio UI launched

### Option 2: Python Script
Convert the notebook to `.py` or use as is:
```bash
python code_markdown.py
```

## How It Works (Summary)
- The model is lazily downloaded if not already present in the `models/` directory.
- The `.env` file is loaded to retrieve the Hugging Face token.
- `llama-cpp-python` loads the GGUF model in CPU-only mode (ideal for Mac M1).
- A code file is read from disk and passed into a structured prompt.
- The LLM generates a detailed explanation which is stored in `code_explanation.txt`.
- This explanation is embedded and saved into ChromaDB.
- When a user asks a question, the closest context is retrieved using cosine similarity.
- A final prompt is built from the question + context and sent back to the model.
- Answer is generated and displayed.
- The Gradio UI allows you to enter any code and ask related questions interactively.

## Limitations
- Only tested for local execution on macOS (Apple Silicon)
- Single file at a time (multi-file traversal can be added)
- No CUDA/GPU support as of now — to be added in future
- Model load time (~10–20s) on first run depending on RAM

## License
MIT License

## Coming Soon
- Multi-file folder traversal with auto vectorization
- Web UI with history + chat memory
- CUDA + Windows/Linux support
- CI/CD enabled Dockerized deployment

---

For questions or improvements, feel free to raise an issue or submit a pull request.
