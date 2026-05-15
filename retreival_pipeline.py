from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
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

def retrieve_documents(query, vector_store, k=5):
    """Retrieve documents from the vector store based on the query"""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.3, 
            "k": k}
    )
    documents = retriever.invoke(query)
    return documents

def generate_answer(query, documents):
    if not documents:
        return "I don't have enough information to answer the question based on the provided documents"
    
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
    
    prompt_template = """Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say I don't have enough information to answer the question based on the provided documents.

Documents:
{context}

Question: {query}
Answer:"""

    prompt = PromptTemplate.from_template(prompt_template)
    
    # Initialize Ollama LLM (you can change the model if you prefer another)
    llm = ChatOllama(model="gemma3:4b")
    
    chain = prompt | llm
    
    print("\nGenerating polished answer with LLM...")
    response = chain.invoke({"context": context, "query": query})
    return response.content

def main():
    #Load vector store
    vector_store = load_vector_store()

    #Retrieve documents
    query = "an example of a postpositive adjective?"
    documents = retrieve_documents(query, vector_store)

    #Print retrieved documents summary
    print(f"\nRetrieved {len(documents)} documents.")
    
    #Generate final answer
    answer = generate_answer(query, documents)
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(answer)

if __name__ == "__main__":
    main()