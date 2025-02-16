import os
import re
import requests
import numpy as np
import gradio as gr

# ---- LANGCHAIN IMPORTS & CACHING SETUP ----
import langchain
from langchain.cache import InMemoryCache
from langchain.llms.base import LLM
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

# Activate LangChain's in-memory cache
langchain.llm_cache = InMemoryCache()


# --------------------------------------------------------------------------------
# 1. Define a custom LLM class that calls our local endpoint & uses LangChain cache
# --------------------------------------------------------------------------------
class LocalLLM(LLM):
    """A custom LLM class that communicates with a local LLM endpoint for chat completions.
    
    We use typed class attributes so that Pydantic (which underlies LangChain's LLM base) 
    recognizes these as valid fields.
    """

    model_name: str = "llama-3.2-3b-instruct"
    temperature: float = 0.0
    endpoint: str = "http://localhost:1234/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "local_llm"

    def _call(self, prompt: str, stop=None) -> str:
        """Make a POST request to the local LLM endpoint with the prompt."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer based on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": -1
        }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request failed
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> dict:
        """
        Helps LangChain caching to determine when prompts+params are the same.
        Return any fields that uniquely identify this LLM's settings.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "endpoint": self.endpoint
        }


# Instantiate our custom local LLM (now recognized by Pydantic)
local_llm = LocalLLM()


# --------------------------------------------------------------------------
# 2. Define helper function to call the LLM with a custom system message
# --------------------------------------------------------------------------
def call_llm(system_message: str, user_message: str) -> str:
    """
    Combines system and user messages into a single prompt string,
    then calls local_llm (which is cached by LangChain).
    """
    combined_prompt = f"{system_message}\n\n{user_message}"
    return local_llm(combined_prompt)


# --------------------------------------------
# 3. Adaptive PDF Chunking
# --------------------------------------------
def load_and_adaptive_chunk_pdf(pdf_path: str, max_chunk_size=1000) -> list[Document]:
    """
    Loads the PDF and splits its pages into paragraphs. Then, it groups paragraphs
    up to a maximum chunk size (in characters).
    """
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()  # Typically one Document per PDF page

    final_docs = []
    for doc in pdf_docs:
        # Split the page by double newlines => approximate paragraphs
        paragraphs = doc.page_content.split("\n\n")
        current_chunk = ""
        current_metadata = doc.metadata.copy()

        for paragraph in paragraphs:
            paragraph_stripped = paragraph.strip()
            if not paragraph_stripped:
                # Skip empty paragraphs
                continue

            # If adding this paragraph to the current chunk won't exceed max_chunk_size, keep adding
            if len(current_chunk) + len(paragraph_stripped) < max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph_stripped
                else:
                    current_chunk = paragraph_stripped
            else:
                # Store the current chunk as a separate Document
                final_docs.append(Document(page_content=current_chunk, metadata=current_metadata))
                # Start a new chunk with the current paragraph
                current_chunk = paragraph_stripped

        # If there's any leftover text in current_chunk after the loop, add it
        if current_chunk:
            final_docs.append(Document(page_content=current_chunk, metadata=current_metadata))
            current_chunk = ""

    return final_docs


# ------------------------------------------------
# 4. Summarization helper
# ------------------------------------------------
def summarize_text(long_text: str) -> str:
    """
    Calls the LLM to summarize a long text.
    """
    system_msg = "You are a summarization assistant."
    summarize_prompt = f"Summarize this text:\n{long_text}\nSummary:"
    return call_llm(system_msg, summarize_prompt)


def summarize_context_if_needed(docs: list[Document], length_threshold=1000) -> list[Document]:
    """
    If a document chunk is longer than 'length_threshold' characters, we summarize it.
    Otherwise, we keep it as is.
    """
    summarized_docs = []
    for doc in docs:
        if len(doc.page_content) > length_threshold:
            summary = summarize_text(doc.page_content)
            summarized_docs.append(Document(page_content=summary, metadata=doc.metadata))
        else:
            summarized_docs.append(doc)
    return summarized_docs


# ------------------------------------------------
# 5. Create embeddings + vector store
# ------------------------------------------------
def create_vector_store(docs: list[Document]) -> FAISS:
    """
    Converts documents into embeddings and stores them in a FAISS index.
    """
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


# ------------------------------------------------
# 6. Expand & Optimize Query
# ------------------------------------------------
def expand_and_optimize_query(user_query: str) -> str:
    """
    Expands the user's query by calling the LLM to add synonyms or related terms.
    """
    system_msg = "Expand the user's query with synonyms and related terms."
    rewrite_prompt = f"Rewrite: {user_query}"
    return call_llm(system_msg, rewrite_prompt)


# ------------------------------------------------
# 7. Vector retrieval with a score threshold
# ------------------------------------------------
def retrieve_with_threshold(vector_store: FAISS, query: str, k=5, threshold=0.2) -> list[Document]:
    """
    Retrieves up to 'k' documents from the vector store, filtering out
    any with a similarity score below 'threshold'.
    """
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    # docs_and_scores is a list of (Document, score)
    return [doc for (doc, score) in docs_and_scores if score >= threshold]


# ------------------------------------------------
# 8. LLM-based re-ranking
# ------------------------------------------------
def llm_based_rerank(docs: list[Document], query: str, top_n=3) -> list[Document]:
    """
    Calls the LLM to assign a numeric relevance score to each document.
    Then returns the top_n most relevant.
    """
    scored_docs = []
    for doc in docs:
        system_msg = "You are a relevance rating assistant."
        user_prompt = (
            f"Query: {query}\n\n"
            f"Document:\n{doc.page_content}\n\n"
            "On a scale of 0-10, how relevant is this document to the query?"
        )

        score_response = call_llm(system_msg, user_prompt)
        # Attempt to parse a number from the response
        matches = re.findall(r"\d+", score_response)
        if matches:
            # Take the first numeric match
            score = float(matches[0])
        else:
            score = 0.0  # fallback if no number found

        scored_docs.append((doc, score))

    # Sort by descending score and take the top_n
    top_docs = sorted(scored_docs, key=lambda x: -x[1])[:top_n]
    return [doc for doc, _ in top_docs]


# ------------------------------------------------
# 9. Build the final prompt for the LLM
# ------------------------------------------------
def build_prompt(user_query: str, docs: list[Document]) -> str:
    """
    Constructs the final prompt for the LLM to answer the user's query,
    including the relevant context from selected docs.
    """
    system_message = (
        "Answer using ONLY the provided context unless it a greeting answer normally and say how can i help you :) . Cite sources as (Page X)."
    )
    context_section = "CONTEXT:\n"
    for doc in docs:
        # +1 in case your PDF loader stores zero-based page indices
        page_num = doc.metadata.get('page', 0) + 1
        context_section += f"Page {page_num}:\n{doc.page_content}\n\n"

    return f"{system_message}\n\n{context_section}\nQuestion: {user_query}\nAnswer:"



def advanced_rag_pipeline(pdf_path: str, user_query: str) -> str:
    # 1. Adaptive chunking of the PDF
    docs = load_and_adaptive_chunk_pdf(pdf_path, max_chunk_size=1000)

    # 2. Create a vector store for retrieval
    vector_store = create_vector_store(docs)

    # 3. Expand the query
    expanded_query = expand_and_optimize_query(user_query)

    # 4. Retrieve top matches (filtering low-similarity docs)
    retrieved_docs = retrieve_with_threshold(vector_store, expanded_query)

    # 5. Summarize any very long docs
    summarized_docs = summarize_context_if_needed(retrieved_docs, length_threshold=1000)

    # 6. Re-rank those docs with the LLM
    best_docs = llm_based_rerank(summarized_docs, expanded_query, top_n=3)

    # 7. Build final prompt and request an answer
    final_prompt = build_prompt(user_query, best_docs)
    answer = call_llm("Answer the user's query using only the provided context.", final_prompt)
    return answer



def gradio_chatbot_demo(pdf_path: str):
    def chat_fn(history, query):
        response = advanced_rag_pipeline(pdf_path, query)
        history.append((query, response))
        return history, ""
    
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Chatbot (Page Number Citations) with Caching & Adaptive Chunking")
        chatbot = gr.Chatbot()
        query_box = gr.Textbox(label="Ask about the PDF")
        query_box.submit(chat_fn, [chatbot, query_box], [chatbot, query_box])
    demo.launch()



def gradio_chatbot_demo(pdf_path: str):
    def chat_fn(history, query):
        response = advanced_rag_pipeline(pdf_path, query)
        history.append((query, response))
        return history, ""
    
    with gr.Blocks() as demo:
        gr.Markdown("# PDF Chatbot (Page Number Citations) with Caching & Adaptive Chunking")
        chatbot = gr.Chatbot()
        query_box = gr.Textbox(label="Ask about the PDF")
        # Use submit method to update the chatbot based on user input
        query_box.submit(chat_fn, [chatbot, query_box], [chatbot, query_box])
    demo.launch()

if __name__ == "__main__":
    PDF_PATH = "Umbrella Corporation Employee Handbook.pdf"
    gradio_chatbot_demo(PDF_PATH)
