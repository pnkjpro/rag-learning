import os
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path: str):
    """Load all pdf files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} not found.")
    
    #Load all pdf files
    loader = DirectoryLoader(
        docs_path, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
        )
    
    documents = loader.load()
    if(len(documents) == 0):
        raise FileNotFoundError(f"No PDF files found in {docs_path}")

    print(f"Loaded {len(documents)} documents from {docs_path}")
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Splitting documents into smaller chunks with overlap"""
    print(f"Splitting documents into {chunk_size} character chunks with {chunk_overlap} overlap...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} chars")
            print(f"Content:")
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_embeddings_and_store(chunks, persist_dir="db/chroma_db"):
    """Create embeddings for chunks and store them in Chroma DB"""
    print("Creating embeddings and storing them in Chroma DB...")

    #Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text" # You can change this to any embedding model you have pulled in Ollama (e.g., mxbai-embed-large)
    )

    #Create Chroma DB vector store with cosine similarity
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Process chunks in batches with progress bar
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)

    print(f"Embeddings created and stored in {persist_dir}")
    return vector_store

def main():
    #Load documents
    documents = load_documents(docs_path="docs")

    #Split documents
    chunks = split_documents(documents)

    #Create embeddings and store in Chroma DB
    vector_store = create_embeddings_and_store(chunks)

    
if __name__ == "__main__":
    main()