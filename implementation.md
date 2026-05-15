# Ingestion Pipeline Implementation Changes

This document tracks the updates and improvements made to the `ingestion_pipeline.py` script.

## 1. Added Python Execution Entry Point
**Change:** Appended `if __name__ == "__main__": main()` to the bottom of the script.
**Reason:** The script originally defined the `main()` function but lacked the standard Python execution block to actually call it. This caused the script to exit silently without running any of the pipeline logic when executed via `python ingestion_pipeline.py`.

## 2. Migrated to Local Embeddings (Ollama)
**Change:** Replaced `OpenAIEmbeddings` with `OllamaEmbeddings` using the `nomic-embed-text` model.
**Reason:** To remove the dependency on paid external APIs (OpenAI) and utilize local inference. This prevents `Missing credentials` errors for users who don't have an `OPENAI_API_KEY` set up, keeping the stack fully open-source and locally executable.

## 3. Improved Document Chunking strategy
**Change:** Switched from `CharacterTextSplitter` to `RecursiveCharacterTextSplitter`.
**Reason:** The original `CharacterTextSplitter` was strictly splitting on double-newlines (`\n\n`). When a PDF page did not contain double-newlines, it kept the entire page text as a single massive chunk (often exceeding 4,000 characters). These oversized chunks exceeded the context window limitations of the local embedding model, resulting in HTTP 500 errors (`"the input length exceeds the context length"`).
`RecursiveCharacterTextSplitter` fixes this by recursively applying a hierarchy of separators (`\n\n`, `\n`, ` `, `""`) until it successfully breaks the text down to the requested chunk size limit (800 characters), ensuring smooth embedding generation.

## 4. Batched Embedding Generation & Progress Tracking
**Change:** Implemented a batching loop (`batch_size = 100`) with a `tqdm` progress bar instead of using `Chroma.from_documents()` directly.
**Reason:** Generating embeddings locally takes time. `Chroma.from_documents()` processes the entire dataset at once, which provides zero feedback to the user on progress and looks like the process is frozen (hanging). By splitting the documents into batches of 100 and utilizing `tqdm`, the user now receives real-time progress updates with estimated time remaining and prevents memory/resource bottlenecking on the local machine.
