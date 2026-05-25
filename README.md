# ⚡ Context-Aware RAG Matrix Lab

## 🧠 Objective of the Project

The objective of this project is to build a **context-aware Retrieval-Augmented Generation (RAG) system** using a lightweight architecture without external frameworks like LangChain.

The system demonstrates how **LLMs can be enhanced with local knowledge retrieval** to answer domain-specific questions more accurately by injecting relevant context at runtime.

Key goals include:

- Implementing a simple RAG pipeline from scratch
- Using keyword-based retrieval instead of vector databases
- Enhancing LLM responses using dynamically retrieved context
- Maintaining conversational memory within a Streamlit interface
- Building an interactive AI assistant powered by Groq LLMs

---

## ⚙️ Methodology / Approach

### 1. Knowledge Base Construction
A local file (`knowledge.txt`) acts as the system's internal knowledge source.

- The file is loaded at runtime
- Text is split into logical chunks (paragraph-based segmentation)
- Each chunk represents an independent knowledge unit

---

### 2. Context Retrieval (Lightweight RAG Engine)

Instead of using embeddings or vector databases, a **keyword matching algorithm** is used:

- User query is tokenized
- Each knowledge chunk is scored based on keyword overlap
- Top matching chunks are selected
- Relevant context is injected into the prompt

This makes the system:
- Fast
- Lightweight
- Easy to understand and modify

---

### 3. Context Injection Strategy

The retrieved context is injected into the LLM prompt as:

- A system-level message
- Clearly separated from conversation history
- Ensuring the model prioritizes internal knowledge

This allows the LLM to:
- Ground responses in provided data
- Avoid hallucinations
- Improve factual consistency

---

### 4. Conversational Memory System

The application maintains chat history using `st.session_state`:

- Stores full conversation (user + assistant messages)
- Preserves system instructions
- Enables multi-turn reasoning

---

### 5. LLM Integration (Groq API)

The system uses:

- **Model:** Llama 3.3 70B Versatile  
- **Provider:** Groq API  
- **Temperature:** 0.2 (for balanced creativity + accuracy)

The model generates responses based on:
- Chat history
- Retrieved context
- Current user query

---

### 6. Streamlit UI Design

The interface is built using Streamlit with custom styling:

- Glassmorphic UI design
- Sidebar control panel
- Live telemetry display
- Chat-based interaction system
- Real-time response streaming

---

## 🚀 Key Features

- ⚡ Lightweight RAG pipeline (no LangChain / no vector DB)
- 📄 Local document-based knowledge retrieval
- 🧠 Context-aware LLM responses
- 💬 Multi-turn conversational memory
- 🎯 Keyword-based retrieval engine
- 🎨 Modern Streamlit UI with custom CSS
- 📊 Live system telemetry and context tracking

---

## 📊 Key Results / Observations

### 1. Effective Context Injection
Injecting retrieved context significantly improves answer relevance and reduces hallucination.

---

### 2. Lightweight Retrieval Works Surprisingly Well
Even without embeddings, keyword-based retrieval performs well for structured or semi-structured knowledge bases.

---

### 3. Conversational Memory Enhances Coherence
Session-based memory allows the system to maintain context across multiple turns, improving user experience.

---

### 4. LLM Performance (Groq + Llama 3.3)
- Fast inference due to Groq acceleration
- High-quality reasoning with minimal latency
- Stable performance in structured QA tasks

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Groq API**
- **Llama 3.3 70B**
- **Regex-based retrieval engine**
- **Dotenv (.env) for API key management**

---

## 📁 Project Structure

```
├── app.py
├── knowledge.txt
├── .env

```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install streamlit groq python-dotenv
```

---

### 2. Set API key in `.env`
```bash
GROQ_API_KEY=your_api_key_here
```

---

### 3. Run the app
```bash
streamlit run app.py
```

---

### 4. Open in browser
```
http://localhost:8501
```

---

## 🎯 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Prompt engineering
- Context injection strategies
- LLM integration (Groq API)
- Streamlit UI development
- Lightweight information retrieval systems
- Conversational AI design
- Real-time AI application engineering

---

## 📌 Future Improvements

- Replace keyword retrieval with embedding-based vector search
- Add FAISS or ChromaDB for scalable retrieval
- Improve chunking using semantic segmentation
- Add document upload support (PDF/CSV)
- Implement evaluation metrics for RAG accuracy

---

## 🧠 Summary

This project demonstrates how a **minimal RAG architecture** can be built without complex frameworks while still achieving strong context-aware AI responses using modern LLMs.

It bridges the gap between:
> simple keyword search systems  
and  
> full-scale production RAG pipelines
