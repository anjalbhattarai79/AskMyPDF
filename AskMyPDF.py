import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AskMyDocs", page_icon="📄")
st.title("📄 AskMyDocs: Chat with Your PDF")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load & split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Vector Store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="my_chroma_db",
        collection_name="pdf_collection"
    )
    vector_store.add_documents(chunks)

    st.session_state.vector_store = vector_store
    st.success("✅ PDF processed and ready!")

# --- Chat Section ---
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided PDF context.
        If the context is insufficient, just say you don't know.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    # --- Runnable Chain ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    st.subheader("💬 Ask your PDF")
    user_q = st.text_input("Enter your question")
    if st.button("Get Answer") and user_q:
        with st.spinner("Thinking..."):
            response = chain.invoke(user_q)
        st.markdown(f"**Answer:** {response}")
