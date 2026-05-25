import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

# Custom CSS
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

# Header
st.markdown("""
    <div class="title-box">
        <h1 style="color:white; margin:0;">🏢 HR Policy Assistant</h1>
        <p style="color:#cce0ff; margin:8px 0 0 0;">Ask anything about company HR policies</p>
    </div>
""", unsafe_allow_html=True)

# Sample questions
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

# Input
question = st.text_input(
    "🔍 Ask your HR question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How many casual leaves am I entitled to?"
)

# Button
if st.button("Ask", use_container_width=True, type="primary"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching HR policy..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question}
                )
                data = response.json()

                # Answer
                st.markdown("### 📋 Answer")
                st.markdown(f"""
                    <div class="answer-box">
                        {data['answer']}
                    </div>
                """, unsafe_allow_html=True)

                # Sources
                st.markdown("### 📄 Sources")
                for i, source in enumerate(data['sources']):
                    st.markdown(f"""
                        <div class="source-box">
                            <b>📌 Source {i+1} — Page {source['page'] + 1}</b><br><br>
                            {source['content']}...
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"API Error: {str(e)} — Make sure FastAPI server is running.")

# Footer
st.markdown("---")
st.markdown("""
    <p style="text-align:center; color:#888; font-size:13px;">
        Powered by RAG • LangChain • Groq LLaMA • FAISS
    </p>
""", unsafe_allow_html=True)