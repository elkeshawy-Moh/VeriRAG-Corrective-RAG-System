"""
app.py
Gradio Web Interface for VeriRAG: Self-Correcting Document Intelligence System.
"""

import os
import gradio as gr
from dotenv import load_dotenv
from rag_engine import load_and_split_documents, CorrectiveRAGSystem

# Load environment variables from .env file if available
load_dotenv()

# Initialize the Corrective RAG Engine (Users can also enter their key in the UI)
DEFAULT_API_KEY = os.getenv("GROQ_API_KEY", "")


def build_knowledge_base(files, api_key, current_db):
    if not api_key or not api_key.strip():
        return current_db, None, "⚠️ **Error:** Please enter your Groq API Key first."
    if not files:
        return current_db, None, "⚠️ **Status:** Please upload at least one PDF or TXT file."

    try:
        rag_engine = CorrectiveRAGSystem(api_key=api_key.strip())
        file_paths = [f.name for f in files]
        chunks = load_and_split_documents(file_paths)

        if not chunks:
            return current_db, None, "❌ **Error:** No readable text extracted from the files."

        vector_db = rag_engine.build_vector_store(chunks)
        status = (
            f"✅ **Knowledge Base Ready!**\n\n"
            f"- **Files Indexed:** {len(files)}\n"
            f"- **Verified Chunks:** {len(chunks)}\n"
            f"- **Vector Store:** FAISS (all-MiniLM-L6-v2)"
        )
        return vector_db, rag_engine, status
    except Exception as e:
        return current_db, None, f"❌ **Error:** {str(e)}"


def ask_question(question, chat_history, vector_db, rag_engine):
    if not question or not question.strip():
        return "", chat_history, "*Waiting for a question...*", "*No execution trace yet.*"

    if vector_db is None or rag_engine is None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "⚠️ Please enter your API key and build the Knowledge Base first."})
        return "", chat_history, "No documents indexed.", "🛑 Pipeline halted."

    result = rag_engine.query(question, vector_db)
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": result["answer"]})

    return "", chat_history, result["sources"], result["trace"]


def clear_chat():
    return [], "*Sources will appear here.*", "*Pipeline trace will appear here.*"


# Gradio UI Definition
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(theme=theme, title="VeriRAG | Corrective RAG System") as demo:
    vector_db_state = gr.State(value=None)
    rag_engine_state = gr.State(value=None)

    gr.Markdown(
        """
        # 🛡️ VeriRAG: Self-Correcting Document Intelligence System
        **End-to-End Corrective Retrieval-Augmented Generation (CRAG) Pipeline**  
        *Evaluates retrieved context, rewrites weak queries automatically, and delivers hallucination-free answers with sources.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🔑 1. API Configuration")
            api_key_input = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password",
                value=DEFAULT_API_KEY
            )

            gr.Markdown("### 📂 2. Document Ingestion")
            file_uploader = gr.File(
                label="Upload PDF or TXT Files",
                file_types=[".pdf", ".txt"],
                file_count="multiple"
            )
            build_btn = gr.Button("⚙️ Build Knowledge Base", variant="primary", size="lg")
            status_box = gr.Markdown("⏳ **Status:** Waiting for API key & documents...")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 3. Ask Verified Questions")
            chatbot = gr.Chatbot(height=380, type="messages", label="VeriRAG Assistant")

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                    scale=5
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                clear_btn = gr.Button("Clear 🗑️", variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("#### 📌 Verified Sources")
                    sources_box = gr.Markdown("*Sources will appear here.*")
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("#### 🔍 Corrective RAG Trace")
                    trace_box = gr.Markdown("*Pipeline trace will appear here.*")

    build_btn.click(
        fn=build_knowledge_base,
        inputs=[file_uploader, api_key_input, vector_db_state],
        outputs=[vector_db_state, rag_engine_state, status_box]
    )
    send_btn.click(
        fn=ask_question,
        inputs=[question_box, chatbot, vector_db_state, rag_engine_state],
        outputs=[question_box, chatbot, sources_box, trace_box]
    )
    question_box.submit(
        fn=ask_question,
        inputs=[question_box, chatbot, vector_db_state, rag_engine_state],
        outputs=[question_box, chatbot, sources_box, trace_box]
    )
    clear_btn.click(fn=clear_chat, outputs=[chatbot, sources_box, trace_box])

if __name__ == "__main__":
    demo.launch()
