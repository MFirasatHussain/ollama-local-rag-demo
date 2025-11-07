# Local RAG chatbot using Ollama + LangChain + Chroma
# Usage: python local_rag_ollama.py yourfile.pdf "your question"

import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def build_rag_pipeline(pdf_path):
    text = load_pdf(pdf_path)
    print(f"Loaded {len(text)} characters from {pdf_path}")
    
    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)
    
    # Create local embeddings & vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_texts(docs, embeddings)
    
    # Create QA chain
    llm = Ollama(model="gemma2:2b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python local_rag_ollama.py <pdf_path> <question>")
        sys.exit(1)
    
    pdf_path, question = sys.argv[1], sys.argv[2]
    qa = build_rag_pipeline(pdf_path)
    print("\nAnswer:\n", qa.run(question))