# Ollama Local RAG Demo — GTA Vice City Edition

A lightweight Retrieval-Augmented Generation (RAG) demo built with [Ollama](https://ollama.com) and [LangChain](https://www.langchain.com/), running entirely offline on your local machine.

Instead of research papers or technical documents, this setup uses a PDF of GTA Vice City cheat codes and lets you query it naturally, for example:
> "Which cheat code gives me a tank?"  
> "What does PANZER do?"

Everything, from embeddings to generation, runs locally using Gemma 2B and nomic-embed-text.

---

## Tech Stack
- Ollama for local LLM inference (gemma2:2b)
- LangChain for retrieval orchestration
- Chroma as the vector store
- PyPDF2 for PDF ingestion
- Python as the main runtime

---

## Installation & Setup

### 1. Install Ollama
Download and install Ollama for your operating system:
https://ollama.com/download

After installation, verify it works:
```bash
ollama --version

Pull the main language model and the embedding model:

ollama pull gemma2:2b
ollama pull nomic-embed-text