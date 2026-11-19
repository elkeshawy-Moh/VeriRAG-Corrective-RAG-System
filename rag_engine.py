"""
rag_engine.py
Core Document Ingestion, Vector Store Indexing, and Corrective RAG (CRAG) Pipeline.
"""

import os
import re
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def clean_text(text: str) -> str:
    """Cleans raw extracted text by normalizing whitespace and newlines."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def load_and_split_documents(file_paths: list[str]) -> list[Document]:
    """Extracts text from PDF/TXT files, attaches metadata, and splits into chunks."""
    raw_documents = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        if file_path.lower().endswith(".pdf"):
            pdf_doc = pymupdf.open(file_path)
            for page_num in range(len(pdf_doc)):
                cleaned_page = clean_text(pdf_doc[page_num].get_text("text"))
                if cleaned_page:
                    raw_documents.append(
                        Document(
                            page_content=cleaned_page,
                            metadata={"source": file_name, "page": page_num + 1}
                        )
                    )
            pdf_doc.close()

        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                cleaned_txt = clean_text(f.read())
                if cleaned_txt:
                    raw_documents.append(
                        Document(
                            page_content=cleaned_txt,
                            metadata={"source": file_name, "page": 1}
                        )
                    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(raw_documents)


class CorrectiveRAGSystem:
    """End-to-end Corrective RAG Engine with relevance grading and query rewriting."""

    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=1024
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self._init_chains()

    def _init_chains(self):
        # 1. Relevance Grader Chain
        grader_prompt = ChatPromptTemplate.from_template(
            """You are a strict relevance evaluator for a RAG system.
Determine if the retrieved document chunk contains useful information to answer the user's question.

User Question: {question}
Retrieved Chunk: {context}

Reply with ONLY 'YES' if relevant, or ONLY 'NO' if off-topic or weak.
Decision:"""
        )
        self.grader_chain = grader_prompt | self.llm | StrOutputParser()

        # 2. Query Rewriter Chain
        rewriter_prompt = ChatPromptTemplate.from_template(
            """You are a search query optimization expert.
Initial retrieval failed to find relevant information. Rewrite the question to make it clearer and better suited for semantic vector search.

Original Question: {question}
Output ONLY the rewritten query without prefixes or quotes:"""
        )
        self.rewriter_chain = rewriter_prompt | self.llm | StrOutputParser()

        # 3. Grounded Answer Generator Chain
        generator_prompt = ChatPromptTemplate.from_template(
            """You are VeriRAG, a reliable document assistant.
Answer the user's question using ONLY the verified context provided below.

Verified Context:
{context}

User Question: {question}

Rules:
1. Rely ONLY on facts explicitly stated in the Verified Context.
2. Do NOT use outside knowledge or make assumptions.
3. Answer in the same language as the user's question (Arabic or English).

Final Answer:"""
        )
        self.generator_chain = generator_prompt | self.llm | StrOutputParser()

    def build_vector_store(self, chunks: list[Document]) -> FAISS:
        """Creates a FAISS vector database from processed document chunks."""
        return FAISS.from_documents(documents=chunks, embedding=self.embedding_model)

    def evaluate_chunks(self, question: str, chunks: list[Document]) -> list[Document]:
        """Filters retrieved chunks, keeping only those graded as 'YES'."""
        verified = []
        for chunk in chunks:
            decision = self.grader_chain.invoke({
                "question": question,
                "context": chunk.page_content
            }).strip().upper()
            if "YES" in decision:
                verified.append(chunk)
        return verified

    def query(self, question: str, vector_db: FAISS) -> dict:
        """Executes Retrieval -> Evaluation -> Query Rewrite (if needed) -> Grounded Generation."""
        trace_logs = ["🔍 **Step 1 (Initial Retrieval):** Searching FAISS vector store..."]
        initial_chunks = vector_db.similarity_search(question, k=4)

        verified_chunks = self.evaluate_chunks(question, initial_chunks)
        trace_logs.append(f"⚖️ **Step 2 (Relevance Evaluation):** {len(verified_chunks)}/{len(initial_chunks)} chunks passed grading.")

        # Corrective Action if initial context is weak or irrelevant
        if len(verified_chunks) == 0:
            rewritten_query = self.rewriter_chain.invoke({"question": question}).strip()
            trace_logs.append(f"🔄 **Step 3 (Query Correction Triggered):** Rewrote query to *'{rewritten_query}'*")
            second_chunks = vector_db.similarity_search(rewritten_query, k=4)
            verified_chunks = self.evaluate_chunks(rewritten_query, second_chunks)
            trace_logs.append(f"⚖️ **Step 4 (Re-Evaluation):** {len(verified_chunks)}/{len(second_chunks)} chunks passed after correction.")
        else:
            trace_logs.append("✅ **Step 3 (Status):** Initial context is relevant. No query rewrite needed.")

        # Handle Unanswered Questions safely
        if len(verified_chunks) == 0:
            trace_logs.append("🛑 **Final Decision:** No verified context found. Declining answer to prevent hallucination.")
            return {
                "answer": "⚠️ عذراً، لم أتمكن من العثور على إجابة موثقة لهذا السؤال داخل المستندات المرفوعة.\n\n*(Sorry, I could not find verified information in the uploaded documents to answer this question.)*",
                "sources": "No verified sources found.",
                "trace": "\n".join(trace_logs)
            }

        # Format context and unique sources
        context_text = "\n\n---\n\n".join([doc.page_content for doc in verified_chunks])
        unique_sources = []
        for doc in verified_chunks:
            label = f"📄 **{doc.metadata.get('source', 'Unknown')}** (Page {doc.metadata.get('page', 1)})"
            if label not in unique_sources:
                unique_sources.append(label)

        trace_logs.append("✍️ **Final Step (Grounded Generation):** Generating answer strictly from verified chunks...")
        final_answer = self.generator_chain.invoke({"context": context_text, "question": question})

        return {
            "answer": final_answer,
            "sources": "\n".join(unique_sources),
            "trace": "\n".join(trace_logs)
        }
