# AskMyPDF 📄🤖

A Retrieval-Augmented Generation (RAG) based application that lets you **chat with your PDFs**.  
Upload a document, ask questions, and get precise answers based on its content.

Deployed here: [AskMyPDF on Render](https://askmypdf-jxda.onrender.com/)

---

## 🚀 User Flow
1. **Upload PDF** – Choose any document you want to explore.  
2. **Ask Questions** – Type in your query about the uploaded PDF.  
3. **Get Answers** – The system retrieves relevant document sections and generates accurate answers using LLMs.  

---

## ⚙️ Technical Flow
1. **PDF Upload & Processing**
   - Doc loader: `PyPDFLoader`  
     (from `langchain.document_loaders import PyPDFLoader`) — loads PDF pages into documents.
   - Text splitter: `RecursiveCharacterTextSplitter`  
     (from `langchain.text_splitter import RecursiveCharacterTextSplitter`) — splits text into chunks for embedding.

2. **Embedding & Storage**
   - Embedding model: `GoogleGenerativeAIEmbeddings`  
     (from `langchain.embeddings import GoogleGenerativeAIEmbeddings`) — converts chunks to vectors.
   - Vector database: `Chroma`  
     (from `langchain.vectorstores import Chroma`) — stores embeddings (e.g., persisted to `my_chroma_db`).

3. **Retrieval**
   - Use the Chroma vector store as a retriever (e.g., `vector_store.as_retriever()`).

4. **Answer Generation**
   - Chat model: `ChatGoogleGenerativeAI`  
     (from `langchain.chat_models import ChatGoogleGenerativeAI`) — Gemini-backed model used to generate answers from retrieved context.
   - Chain: Retrieval-style chain (e.g., `RetrievalQA` or a custom runnable chain) that formats retrieved docs and calls the chat model.


5. **Frontend**  
   - Built with **Streamlit** for a simple and interactive chat-like interface.  

---

## 🛠️ Tech Stack
- **Python**  
- **LangChain**  
- **Streamlit**  
- **ChromaDB**  
- **Google Generative AI (Gemini)**  

---

## 📌 Future Improvements
- Multi-file support  
- Persistent chat history  
- Support for additional file formats (DOCX, TXT, etc.)  

---