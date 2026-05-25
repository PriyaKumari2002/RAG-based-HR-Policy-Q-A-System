import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ── ENV ──────────────────────────────────────────
load_dotenv()

# ── Page Config ──────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

# ── CSS ──────────────────────────────────────────
st.markdown("""
    <style>
    .answer-box {
        background-color: #1e3a5f;
        border-left: 5px solid #4A90E2;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #ffffff;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .source-box {
        background-color: #2a2a3d;
        border-left: 4px solid #7eb8f7;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 8px 0;
        font-size: 13px;
        color: #d0e4ff;
    }
    .title-box {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #1a3c6e, #4A90E2);
        border-radius: 15px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────
st.markdown("""
    <div class="title-box">
        <h1 style="color:white; margin:0;">🏢 HR Policy Assistant</h1>
        <p style="color:#cce0ff; margin:8px 0 0 0;">Ask anything about company HR policies</p>
    </div>
""", unsafe_allow_html=True)

# ── Pipeline ─────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    with st.spinner("Loading HR Policy documents..."):
        loader = PyPDFLoader("data/hr_policy.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=st.secrets["GROQ_API_KEY"]
        )

        prompt_template = """
You are an expert HR Policy Assistant for an organization.
Your job is to answer employee questions accurately based ONLY on the HR policy document provided.

Instructions:
- Answer ONLY from the context provided below
- Be concise and clear
- If exact information is available, give a direct answer
- If partial information is available, share what you know and mention it may be incomplete
- If no information is available, say "This topic is not covered in the current HR policy document"
- Always mention which section or page the information comes from if possible
- Do not make up any information

Context:
{context}

Employee Question: {question}

HR Assistant Answer:
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    return qa_chain

qa_chain = initialize_pipeline()

# ── Sample Questions ──────────────────────────────
st.markdown("#### 💡 Try asking:")
col1, col2 = st.columns(2)
with col1:
    if st.button("🤒 Sick leave policy?"):
        st.session_state.question = "How many sick leaves are allowed?"
    if st.button("👶 Maternity leave?"):
        st.session_state.question = "What is the maternity leave policy?"
with col2:
    if st.button("📝 Notice period?"):
        st.session_state.question = "What is the notice period for resignation?"
    if st.button("⚖️ Code of conduct?"):
        st.session_state.question = "What is the code of conduct policy?"

st.markdown("---")

# ── Input ────────────────────────────────────────
question = st.text_input(
    "🔍 Ask your HR question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How many casual leaves am I entitled to?"
)

# ── Ask Button ───────────────────────────────────
if st.button("Ask", use_container_width=True, type="primary"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching HR policy..."):
            try:
                response = qa_chain.invoke({"query": question})

                st.markdown("### 📋 Answer")
                st.markdown(f"""
                    <div class="answer-box">
                        {response['result']}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📄 Sources")
                for i, doc in enumerate(response['source_documents']):
                    st.markdown(f"""
                        <div class="source-box">
                            <b>📌 Source {i+1} — Page {doc.metadata['page'] + 1}</b><br><br>
                            {doc.page_content[:200]}...
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ── Footer ───────────────────────────────────────
st.markdown("---")
st.markdown("""
    <p style="text-align:center; color:#888; font-size:13px;">
        Powered by RAG • LangChain • Groq LLaMA 3 • FAISS
    </p>
""", unsafe_allow_html=True)