# RAG

This repository contains a custom implementation of a Retrieval-Augmented Generation (RAG) pipeline for querying PDFs using a local Language Model (LLM) endpoint that was deployed using LM Studio. The project demonstrates adaptive PDF chunking, vector store retrieval, LLM-based re-ranking, and final prompt construction to answer questions using only the provided context from the PDF.

## Features

- **Adaptive PDF Chunking:**  
  Splits PDF pages into paragraphs and groups them into chunks of a specified maximum size. This ensures that each chunk contains contextually coherent information while avoiding overly large inputs.

- **Document Summarization:**  
  Automatically summarizes long document chunks using the LLM, reducing noise and focusing on key information for better retrieval performance.

- **Embeddings & Vector Store:**  
  Converts document chunks into embeddings and stores them in a FAISS index, allowing for efficient similarity search and retrieval of contextually relevant documents.

- **Query Expansion:**  
  Enhances the user's query with synonyms and related terms via the LLM. This expansion improves retrieval recall, especially when the query does not exactly match the language used in the document.

- **LLM-based Re-ranking:**  
  Re-ranks retrieved document chunks by assessing their relevance to the query using the LLM. This additional layer of evaluation ensures that the final context provided to the LLM is of high quality and directly relevant to the query.

- **Context-Aware Answering with Citations:**  
  Constructs a final prompt that includes relevant context along with page citations, ensuring that answers are traceable and verifiable against the source PDF.

- **Interactive Demo with Gradio:**  
  Gradio interface allows for an interactive PDF chatbot experience, making it easy to demo and test the system.

## Why This Approach is Better

1. **Adaptive Chunking vs. Fixed Chunking:**  
   Instead of splitting the PDF into fixed-size chunks or per-page content, this approach uses adaptive chunking based on paragraph boundaries. This results in more coherent context blocks that better capture the semantics of the document.

2. **LLM-based Summarization:**  
   Summarizing overly long chunks reduces the risk of exceeding input size limits for the LLM and ensures that only the most relevant information is processed. This dynamic summarization improves both efficiency and answer quality.

3. **Query Expansion for Improved Recall:**  
   Enhancing the user's query with synonyms and related terms enables the retrieval system to capture a wider array of relevant context, even when there is a vocabulary mismatch between the query and the document.

4. **Re-ranking with LLM:**  
   The use of an LLM to re-rank retrieved documents adds an intelligent layer of filtering, ensuring that only the most relevant context is included in the final prompt. This step leverages the language understanding capabilities of the LLM beyond simple vector similarity scores.

5. **Integrated and Modular Pipeline:**  
   The pipeline is designed to be modular, making it easy to swap out components (e.g., different summarization techniques or vector stores) as needed. This flexibility allows for rapid iteration and customization based on specific use cases.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A local LLM endpoint running (e.g., at `http://localhost:1234/v1/chat/completions`)
- Necessary Python packages (see [Installation](#installation))
