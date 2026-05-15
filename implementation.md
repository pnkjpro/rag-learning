# Ingestion & Retrieval Pipeline Implementation Changes

This document tracks the updates and improvements made to the RAG pipeline scripts.

## 1. Added Python Execution Entry Point
**Change:** Appended `if __name__ == "__main__": main()` to the bottom of the script.
**Reason:** The script originally defined the `main()` function but lacked the standard Python execution block to actually call it. This caused the script to exit silently without running any of the pipeline logic.

## 2. Migrated to Local Embeddings (Ollama)
**Change:** Replaced `OpenAIEmbeddings` with `OllamaEmbeddings` using the `nomic-embed-text` model.
**Reason:** To remove the dependency on paid external APIs (OpenAI) and utilize local inference. This prevents `Missing credentials` errors for users who don't have an `OPENAI_API_KEY` set up.

## 3. Improved Document Chunking strategy
**Change:** Switched from `CharacterTextSplitter` to `RecursiveCharacterTextSplitter`.
**Reason:** The original `CharacterTextSplitter` was strictly splitting on double-newlines (`\n\n`), resulting in massive oversized chunks that exceeded the context window limitations of the local embedding model, resulting in HTTP 500 errors. `RecursiveCharacterTextSplitter` fixes this by recursively applying a hierarchy of separators until it successfully breaks the text down to the requested chunk size limit (800 characters).

## 4. Batched Embedding Generation & Progress Tracking
**Change:** Implemented a batching loop (`batch_size = 100`) with a `tqdm` progress bar instead of using `Chroma.from_documents()` directly.
**Reason:** `Chroma.from_documents()` processes the entire dataset at once, which provides zero feedback to the user on progress and looks like the process is frozen. By splitting the documents into batches of 100 and utilizing `tqdm`, the user now receives real-time progress updates with estimated time remaining and prevents memory/resource bottlenecking on the local machine.

## 5. Score Threshold Filtering
**Change:** Replaced `vector_store.similarity_search` with LangChain's idiomatic `vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6})`.
**Reason:** Instead of returning the top K nearest neighbors regardless of relevancy, the retriever now enforces a strict minimum threshold (like `0.6`). Any chunks scoring below this similarity threshold are discarded, preventing irrelevant information from polluting the final answer.

## 6. LLM Integration & Polished Responses
**Change:** Integrated `ChatOllama` (using `llama3`) and a strict `PromptTemplate` to the retrieval pipeline to generate natural language answers from retrieved chunks.
**Reason:** Instead of just dumping raw text chunks into the console, we now feed the retrieved text as `{context}` into an LLM. We also instituted a strict fallback mechanism: if the vector store returns `0` documents (due to the strict similarity threshold), the pipeline will short-circuit and safely output *"I don't have enough information to answer the question based on the provided documents"*, effectively preventing LLM hallucinations.
