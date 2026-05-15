from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
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

def setup_conversational_rag(vector_store, k=5, threshold=0.3):
    """Setup the history-aware retriever and QA chain"""
    llm = ChatOllama(model="gemma3:4b")
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": threshold, 
            "k": k}
    )
    
    # 1. Contextualize question prompt (for history-aware retrieval)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 2. Answer question prompt (for final generation)
    system_prompt = (
        "Please provide a clear, helpful answer using only the information from the provided documents. "
        "If you can't find the answer in the documents, say I don't have enough information to answer the question based on the provided documents.\n\n"
        "Documents:\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return history_aware_retriever, question_answer_chain

def main():
    # Load vector store
    vector_store = load_vector_store()

    # Setup RAG chains
    history_aware_retriever, question_answer_chain = setup_conversational_rag(vector_store)

    # Initialize conversational history
    chat_history = []
    
    print("\n" + "="*50)
    print("Conversational RAG initialized! Type 'quit' or 'exit' to stop.")
    print("="*50)

    # Interactive CLI Chat Loop
    while True:
        query = input("\nYou: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        # 1. Retrieve relevant documents using history-aware retriever
        retrieved_docs = history_aware_retriever.invoke({
            "input": query, 
            "chat_history": chat_history
        })
        
        # 2. Short-circuit if no relevant docs found (the deterministic fallback)
        if not retrieved_docs:
            answer = "I don't have enough information to answer the question based on the provided documents"
            print(f"AI: {answer}")
        else:
            print(f"[System: Generating polished answer using {len(retrieved_docs)} documents...]")
            # 3. Generate answer using the retrieved documents + history
            answer = question_answer_chain.invoke({
                "input": query, 
                "chat_history": chat_history, 
                "context": retrieved_docs
            })
            print(f"AI: {answer}")
            
        # 4. Update chat history for the next turn
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=answer),
            ]
        )

if __name__ == "__main__":
    main()
