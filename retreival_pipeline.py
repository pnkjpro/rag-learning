from langchain_community.embeddings import OllamaEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_vector_store(persist_dir="db/chroma_db"):
    """Load the vector store from Chroma DB"""
    print(f"Loading vector store from {persist_dir}...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    vector_store = Chroma(  
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store loaded from {persist_dir}")
    return vector_store

def retrieve_documents(query, vector_store, k=4):
    """Retrieve documents from the vector store based on the query"""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.6, 
            "k": k}
    )
    documents = retriever.invoke(query)
    return documents

def main():
    #Load vector store
    vector_store = load_vector_store()

    #Retrieve documents
    query = "an example of a postpositive adjective?"
    documents = retrieve_documents(query, vector_store)

    #Print retrieved documents
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content}")

if __name__ == "__main__":
    main()