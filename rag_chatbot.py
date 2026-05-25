"""
═══════════════════════════════════════════════════════════════════════════════
  ENHANCED RAG CHATBOT  —  Context-Aware Conversational AI
  NO LangChain · Real Vector Embeddings · Cosine Similarity · Groq LLaMA 3.3
═══════════════════════════════════════════════════════════════════════════════

SETUP:
  pip install streamlit groq python-dotenv sentence-transformers numpy pypdf2 docx2txt

.env file:
  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

RUN:
  streamlit run rag_chatbot.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import time
import json
import math
import hashlib
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Optional imports — gracefully degrade if not installed
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME        = "llama-3.3-70b-versatile"
EMBED_MODEL       = "all-MiniLM-L6-v2"          # 80 MB, very fast
CHUNK_SIZE        = 400                           # characters per chunk
CHUNK_OVERLAP     = 80                            # overlap between chunks
TOP_K             = 3                             # chunks to retrieve
MAX_HISTORY_TURNS = 10                            # conversation turns to keep
SIMILARITY_THRESH = 0.25                          # min cosine similarity

DEFAULT_KNOWLEDGE = """\
The Advanced Support Automation Protocol (ASAP-2026) is an internal corporate routing matrix used to triage enterprise customer accounts.
Alpha-Tier failures are handled by senior systems architects within 15 minutes.
Beta-Tier software bugs are handled by regular engineering squads within 2 hours.
Gamma-Tier general inquiries are routed to standard support agents within 24 hours.

The Escalation Override Clause (EOC) allows any customer flagged as "Platinum" status to bypass the standard queue and receive Alpha-Tier treatment regardless of issue severity.
Platinum status is granted after 3 years of continuous contract renewal or annual spend exceeding $500,000.

Customer Satisfaction Index (CSI) scores below 3.0 trigger automatic supervisor review.
A CSI below 2.0 initiates the Emergency Recovery Protocol (ERP), assigning a dedicated Customer Success Manager.

All data is encrypted at rest using AES-256 and in transit using TLS 1.3.
The system undergoes quarterly security audits conducted by third-party firms certified under ISO 27001.

The product lineup includes: NeuralDesk Pro (enterprise helpdesk), FlowSync (workflow automation), and DataBridge (ETL pipeline tool).
NeuralDesk Pro supports up to 500 concurrent agents. FlowSync integrates with over 200 SaaS platforms. DataBridge processes up to 10TB per day.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE  —  Pure NumPy cosine similarity, no external vector DB
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    In-memory vector store using numpy.
    Stores (chunk_text, embedding, metadata) triples.
    Falls back to TF-IDF style keyword scoring when sentence-transformers
    is not installed.
    """

    def __init__(self):
        self.chunks:      List[str]       = []
        self.embeddings:  List[np.ndarray] = []
        self.metadata:    List[Dict]       = []
        self.model:       Optional[object] = None
        self._load_embed_model()

    def _load_embed_model(self):
        if EMBEDDINGS_AVAILABLE:
            with st.spinner("Loading embedding model (first run only)…"):
                self.model = SentenceTransformer(EMBED_MODEL)
        else:
            st.warning("⚠ sentence-transformers not installed — using keyword search fallback.")

    # ── Text chunking ─────────────────────────────────────────────────────────
    @staticmethod
    def chunk_text(text: str, source: str = "unknown") -> List[Tuple[str, Dict]]:
        """Split text into overlapping chunks with metadata."""
        text   = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start  = 0
        idx    = 0
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                chunks.append((chunk, {"source": source, "chunk_id": idx, "char_start": start}))
                idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Embed ─────────────────────────────────────────────────────────────────
    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Fallback: simple TF-IDF-like bag-of-words vector (vocab from stored chunks)
        vocab = self._build_vocab()
        return np.array([self._bow_vector(t, vocab) for t in texts])

    def _build_vocab(self) -> List[str]:
        all_words = set()
        for c in self.chunks:
            all_words.update(re.findall(r'\w+', c.lower()))
        return sorted(all_words)

    @staticmethod
    def _bow_vector(text: str, vocab: List[str]) -> np.ndarray:
        words = set(re.findall(r'\w+', text.lower()))
        vec   = np.array([1.0 if w in words else 0.0 for w in vocab])
        norm  = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── Add documents ─────────────────────────────────────────────────────────
    def add_texts(self, text: str, source: str = "document"):
        raw_chunks = self.chunk_text(text, source)
        if not raw_chunks:
            return 0
        texts = [c[0] for c in raw_chunks]
        metas = [c[1] for c in raw_chunks]
        embeds = self._embed(texts)
        self.chunks.extend(texts)
        self.embeddings.extend(embeds)
        self.metadata.extend(metas)
        return len(texts)

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.chunks:
            return []
        q_emb   = self._embed([query])[0]
        scores  = [self._cosine(q_emb, e) for e in self.embeddings]
        ranked  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score >= SIMILARITY_THRESH:
                results.append({
                    "text":     self.chunks[idx],
                    "score":    round(float(score), 4),
                    "source":   self.metadata[idx].get("source", "unknown"),
                    "chunk_id": self.metadata[idx].get("chunk_id", 0),
                })
        return results

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def stats(self) -> Dict:
        sources = list({m.get("source","?") for m in self.metadata})
        return {"total_chunks": len(self.chunks), "sources": sources}

    def clear(self):
        self.chunks.clear(); self.embeddings.clear(); self.metadata.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY MANAGER  —  Sliding window + summary compression
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Manages conversation history with:
    - Sliding window (keeps last N turns)
    - Token-budget awareness
    - Exportable history
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns   = max_turns
        self.history: List[Dict] = []   # {"role", "content", "timestamp", "ctx_chunks"}
        self.system_prompt = (
            "You are an expert conversational AI assistant with broad knowledge across "
            "technology, science, business, history, and more.\n\n"
            "INSTRUCTIONS:\n"
            "1. If retrieved context is provided and relevant, use it to answer — cite the source.\n"
            "2. If no context is retrieved OR the context is not relevant to the question, "
            "answer using your own training knowledge confidently and thoroughly. "
            "Do NOT say 'I don't have context' for general knowledge questions like "
            "'What is AI?', 'What is ML?', 'Explain Python', etc.\n"
            "3. For questions clearly about the uploaded documents (e.g. internal policies, "
            "ASAP-2026, specific company data), rely on the retrieved context.\n"
            "4. Always use markdown formatting — headings, bullet points, bold text — for clarity.\n"
            "5. Remember the full conversation history and refer to it naturally.\n"
            "6. Be concise but thorough. Never refuse to answer a general knowledge question."
        )

    def add(self, role: str, content: str, ctx_chunks: List[Dict] = None):
        self.history.append({
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "ctx_chunks": ctx_chunks or [],
        })
        # Trim to sliding window
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def build_payload(self, context_str: str, user_query: str) -> List[Dict]:
        """Build the message list for the Groq API."""
        payload = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (exclude last user msg — added below)
        for turn in self.history[:-1]:
            payload.append({"role": turn["role"], "content": turn["content"]})

        # Context injection as a system message
        if context_str.strip():
            payload.append({
                "role": "system",
                "content": (
                    "━━━ RETRIEVED DOCUMENT CONTEXT ━━━\n"
                    f"{context_str}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "The above context is from the user's uploaded documents. "
                    "Use it if it is relevant to the question. "
                    "If it is NOT relevant, ignore it and answer from your own knowledge."
                )
            })

        payload.append({"role": "user", "content": user_query})
        return payload

    def to_export(self) -> str:
        lines = [f"# Conversation Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for turn in self.history:
            lines.append(f"**[{turn['timestamp']}] {turn['role'].upper()}**\n{turn['content']}\n")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len([t for t in self.history if t["role"] == "user"])


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Extract text from uploaded file. Returns (text, source_name)."""
    name = uploaded_file.name
    ext  = name.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore"), name

    elif ext == "pdf":
        if not PDF_AVAILABLE:
            return "", "⚠ PyPDF2 not installed"
        reader = PyPDF2.PdfReader(uploaded_file)
        text   = "\n".join(p.extract_text() or "" for p in reader.pages)
        return text, name

    elif ext in ("docx", "doc"):
        if not DOCX_AVAILABLE:
            return "", "⚠ docx2txt not installed"
        text = docx2txt.process(uploaded_file)
        return text, name

    elif ext == "md":
        return uploaded_file.read().decode("utf-8", errors="ignore"), name

    elif ext == "json":
        data = json.load(uploaded_file)
        return json.dumps(data, indent=2), name

    return "", f"Unsupported format: .{ext}"


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Manrope:wght@400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
}
.stApp {
    background: #05080f;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Header ── */
.rag-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.07) 0%, rgba(0,229,160,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,212,255,0.12), transparent 70%);
    border-radius: 50%;
}
.rag-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #e8edf5, #00d4ff, #00e5a0) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 1.2 !important;
}
.rag-sub {
    color: #5a7290 !important;
    font-size: 0.85rem !important;
    margin-top: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.rag-badges {
    display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;
}
.badge {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
}
.badge-blue  { background:rgba(0,212,255,.1);  color:#00d4ff; border-color:rgba(0,212,255,.3); }
.badge-green { background:rgba(0,229,160,.1);  color:#00e5a0; border-color:rgba(0,229,160,.3); }
.badge-amber { background:rgba(255,179,64,.1); color:#ffb340; border-color:rgba(255,179,64,.3); }
.badge-purple{ background:rgba(167,139,250,.1);color:#a78bfa; border-color:rgba(167,139,250,.3); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080d17 !important;
    border-right: 1px solid #0f1e2f !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 90px;
    background: #0d1420;
    border: 1px solid #1c2a3a;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-lbl { color: #5a7290; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; margin-top: 3px; }

/* ── Chat messages ── */
.stChatMessage {
    background: #0b1221 !important;
    border: 1px solid #0f1e30 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
}
.stChatMessage[data-testid*="user"] {
    border-left: 3px solid #00d4ff !important;
    background: #081525 !important;
}
.stChatMessage[data-testid*="assistant"] {
    border-left: 3px solid #00e5a0 !important;
}

/* ── Chat input ── */
.stChatInput textarea {
    background: #0d1420 !important;
    border: 1px solid #1c2a3a !important;
    border-radius: 12px !important;
    color: #e8edf5 !important;
    font-family: 'Manrope', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.3) !important;
}

/* ── Context expander ── */
.context-chunk {
    background: #08101a;
    border: 1px solid #0f1e30;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #5a7290;
    line-height: 1.6;
}
.context-score {
    display: inline-block;
    background: rgba(0,212,255,.12);
    color: #00d4ff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 6px;
}
.context-source {
    display: inline-block;
    background: rgba(0,229,160,.1);
    color: #00e5a0;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    margin-left: 4px;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #0099cc, #00d4ff) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: .02em !important;
    transition: all .2s !important;
}
.stButton button:hover {
    box-shadow: 0 4px 20px rgba(0,212,255,.35) !important;
    transform: translateY(-1px) !important;
}
.stButton.secondary button {
    background: #0d1420 !important;
    color: #5a7290 !important;
    border: 1px solid #1c2a3a !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #0d1420 !important;
    border: 1px solid #1c2a3a !important;
    border-radius: 10px !important;
    color: #5a7290 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #080d17 !important;
    border: 1px solid #0f1e30 !important;
    border-top: none !important;
}

/* ── Upload zone ── */
.stFileUploader {
    background: #080d17 !important;
    border: 2px dashed #1c2a3a !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: #0f1e30 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: #1c2a3a; border-radius: 3px; }

/* ── Source chips in sidebar ── */
.source-chip {
    display: inline-block;
    background: rgba(167,139,250,.1);
    border: 1px solid rgba(167,139,250,.25);
    color: #a78bfa;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
@st.cache_resource
def get_vector_store():
    """Cached so the embedding model loads only once."""
    vs = VectorStore()
    vs.add_texts(DEFAULT_KNOWLEDGE, source="default_knowledge.txt")
    return vs

if "memory"       not in st.session_state: st.session_state.memory       = ConversationMemory()
if "last_chunks"  not in st.session_state: st.session_state.last_chunks  = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "avg_latency"  not in st.session_state: st.session_state.avg_latency  = []
if "sources_loaded" not in st.session_state: st.session_state.sources_loaded = {"default_knowledge.txt"}

vs     = get_vector_store()
memory: ConversationMemory = st.session_state.memory

# ── API key check ──────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("🔑 **GROQ_API_KEY not found.** Create a `.env` file with `GROQ_API_KEY=gsk_xxx`")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="rag-header">
  <h1 class="rag-title">🧠 RAG Chatbot Lab</h1>
  <p class="rag-sub">Context-Aware Retrieval-Augmented Generation · No LangChain · Pure Python</p>
  <div class="rag-badges">
    <span class="badge badge-blue">Vector Embeddings</span>
    <span class="badge badge-green">Cosine Similarity Search</span>
    <span class="badge badge-amber">Conversation Memory</span>
    <span class="badge badge-purple">Multi-Document RAG</span>
    <span class="badge badge-blue">Groq · LLaMA 3.3-70B</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Knowledge Base Control")

    # ── Stats ──
    vstats = vs.stats()
    avg_ms = int(sum(st.session_state.avg_latency[-20:]) / max(len(st.session_state.avg_latency[-20:]), 1))

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val" style="color:#00d4ff">{vstats['total_chunks']}</div>
        <div class="metric-lbl">Chunks Indexed</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" style="color:#00e5a0">{memory.turn_count}</div>
        <div class="metric-lbl">Turns</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" style="color:#ffb340">{st.session_state.total_queries}</div>
        <div class="metric-lbl">Queries</div>
      </div>
      <div class="metric-card">
        <div class="metric-val" style="color:#a78bfa">{avg_ms}ms</div>
        <div class="metric-lbl">Avg Latency</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Document Upload ──
    st.markdown("#### 📂 Upload Documents")
    st.caption("Supports: `.txt` `.pdf` `.docx` `.md` `.json`")

    uploaded = st.file_uploader(
        "Drop files here",
        type=["txt", "pdf", "docx", "md", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.sources_loaded:
                with st.spinner(f"Indexing {f.name}…"):
                    text, src = parse_uploaded_file(f)
                    if text:
                        n = vs.add_texts(text, source=src)
                        st.session_state.sources_loaded.add(f.name)
                        st.success(f"✅ {f.name} → {n} chunks indexed")
                    else:
                        st.error(f"❌ Could not parse {f.name}")

    # ── Loaded Sources ──
    if st.session_state.sources_loaded:
        st.markdown("**Loaded Sources:**")
        chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in st.session_state.sources_loaded)
        st.markdown(chips, unsafe_allow_html=True)

    st.divider()

    # ── Settings ──
    st.markdown("#### 🎛️ Retrieval Settings")
    top_k      = st.slider("Chunks to retrieve (Top-K)", 1, 8, TOP_K)
    sim_thresh = st.slider("Min similarity threshold", 0.0, 0.9, SIMILARITY_THRESH, 0.05)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)
    show_context = st.toggle("Show retrieved context", value=True)
    show_scores  = st.toggle("Show similarity scores",  value=True)

    st.divider()

    # ── Memory Controls ──
    st.markdown("#### 🧹 Memory Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            memory.clear()
            st.session_state.last_chunks = []
            st.rerun()
    with col_b:
        if st.button("📤 Export", use_container_width=True):
            st.download_button(
                "💾 Download",
                data=memory.to_export(),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if st.button("🔄 Reset Knowledge Base", use_container_width=True):
        vs.clear()
        vs.add_texts(DEFAULT_KNOWLEDGE, source="default_knowledge.txt")
        st.session_state.sources_loaded = {"default_knowledge.txt"}
        memory.clear()
        st.session_state.total_queries = 0
        st.session_state.avg_latency   = []
        st.session_state.last_chunks   = []
        st.success("Knowledge base reset ✓")
        st.rerun()

    st.divider()

    # ── Quick Question Starters ──
    st.markdown("#### 💡 Quick Start Questions")
    starters = [
        "What is Artificial Intelligence?",
        "Explain Machine Learning simply",
        "What is ASAP-2026?",
        "How does the Escalation Override Clause work?",
        "What products are in the lineup?",
        "What is the difference between AI and ML?",
        "What security standards are used?",
    ]
    for q in starters:
        if st.button(q, use_container_width=True, key=f"qs_{q}"):
            st.session_state["prefill_query"] = q
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Render history ─────────────────────────────────────────────────────────────
for turn in memory.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        # Show context chunks that were retrieved for this turn
        if show_context and turn["role"] == "assistant" and turn.get("ctx_chunks"):
            with st.expander(f"🔍 Retrieved Context — {len(turn['ctx_chunks'])} chunk(s)", expanded=False):
                for chunk in turn["ctx_chunks"]:
                    score_html  = f'<span class="context-score">sim: {chunk["score"]:.3f}</span>' if show_scores else ''
                    source_html = f'<span class="context-source">📄 {chunk["source"]}</span>'
                    st.markdown(f'{score_html} {source_html}', unsafe_allow_html=True)
                    st.markdown(f'<div class="context-chunk">{chunk["text"]}</div>', unsafe_allow_html=True)


# ── Chat input ─────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
user_input = st.chat_input(
    "Ask anything about your documents… (Ctrl+Enter to send)",
    key="chat_input",
) or prefill

if user_input:
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    memory.add("user", user_input)

    # 2. Retrieve relevant chunks
    retrieved = vs.search(user_input, top_k=top_k)
    # Filter by dynamic threshold
    retrieved = [r for r in retrieved if r["score"] >= sim_thresh]

    # Build context string
    if retrieved:
        context_str = "\n\n".join(
            f"[Source: {r['source']} | Chunk #{r['chunk_id']} | Score: {r['score']:.3f}]\n{r['text']}"
            for r in retrieved
        )
        retrieval_mode = "📄 Document context used"
    else:
        context_str    = ""   # empty → LLM falls back to its own knowledge
        retrieval_mode = "🧠 LLM general knowledge (no matching docs)"

    # 3. Build payload with memory
    payload = memory.build_payload(context_str, user_input)

    # 4. Call Groq
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        t_start = time.time()

        try:
            stream = groq_client.chat.completions.create(
                messages=payload,
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=1500,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                response_placeholder.markdown(full_response + "▋")

            response_placeholder.markdown(full_response)

            latency_ms = int((time.time() - t_start) * 1000)
            st.session_state.avg_latency.append(latency_ms)
            st.session_state.total_queries += 1

            # Show context in this turn
            if show_context and retrieved:
                with st.expander(f"🔍 Retrieved Context — {len(retrieved)} chunk(s)", expanded=False):
                    for r in retrieved:
                        score_html  = f'<span class="context-score">sim: {r["score"]:.3f}</span>' if show_scores else ''
                        source_html = f'<span class="context-source">📄 {r["source"]}</span>'
                        st.markdown(f'{score_html} {source_html}', unsafe_allow_html=True)
                        st.markdown(f'<div class="context-chunk">{r["text"]}</div>', unsafe_allow_html=True)

            # Latency badge
            st.caption(f"⚡ {latency_ms}ms · {MODEL_NAME} · {retrieval_mode}")

        except Exception as e:
            full_response = f"❌ **API Error:** {str(e)}"
            response_placeholder.markdown(full_response)

    # 5. Save to memory
    memory.add("assistant", full_response, ctx_chunks=retrieved)
    st.session_state.last_chunks = retrieved
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  BOTTOM: RAG PIPELINE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════
if memory.turn_count == 0:
    st.divider()
    st.markdown("#### 🔄 How This RAG Pipeline Works")
    cols = st.columns(5)
    steps = [
        ("📂", "1. Ingest", "Upload TXT, PDF, DOCX, MD files"),
        ("✂️", "2. Chunk", f"Split into {CHUNK_SIZE}-char overlapping segments"),
        ("🧮", "3. Embed", "Sentence-Transformers → 384-dim vectors"),
        ("🔍", "4. Retrieve", f"Cosine similarity → Top-{TOP_K} chunks"),
        ("🤖", "5. Generate", "LLaMA 3.3-70B answers with context"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:#0d1420;border:1px solid #1c2a3a;border-radius:12px;
                        padding:16px;text-align:center;height:130px;">
              <div style="font-size:1.6rem">{icon}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;
                          color:#e8edf5;font-size:.85rem;margin:6px 0 4px">{title}</div>
              <div style="color:#5a7290;font-size:.75rem;line-height:1.4">{desc}</div>
            </div>
            """, unsafe_allow_html=True)