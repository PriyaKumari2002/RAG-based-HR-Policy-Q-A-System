from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ─── Logging Setup ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # file mein save
        logging.StreamHandler()           # terminal mein bhi dikhao
    ]
)
logger = logging.getLogger(__name__)

# ─── ENV Load ────────────────────────────────────────────
load_dotenv(r"C:\Users\kmpri\OneDrive\Desktop\hr-policy-qa\.env")

# ─── FastAPI App ─────────────────────────────────────────
app = FastAPI(
    title="HR Policy Q&A API",
    description="RAG based HR Policy Question Answering System",
    version="1.0.0"
)

# ─── Models ──────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    response_time: float

# ─── Pipeline ────────────────────────────────────────────
def initialize_pipeline():
    try:
        logger.info("Loading PDF...")
        loader = PyPDFLoader(
            r"C:\Users\kmpri\OneDrive\Desktop\hr-policy-qa\data\hr_policy.pdf"
        )
        documents = loader.load()
        logger.info(f"PDF loaded: {len(documents)} pages")

        logger.info("Chunking documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Total chunks: {len(chunks)}")

        logger.info("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        logger.info("Loading FAISS vectorstore...")
        vectorstore = FAISS.load_local(
            r"C:\Users\kmpri\faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded successfully")

        logger.info("Initializing LLM...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
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

        logger.info("Pipeline ready!")
        return qa_chain

    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        raise

# ─── Initialize ──────────────────────────────────────────
qa_chain = initialize_pipeline()

# ─── Routes ──────────────────────────────────────────────
@app.get("/")
def home():
    logger.info("Home endpoint called")
    return {"message": "HR Policy Q&A API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    logger.info(f"Question received: {request.question}")
    start_time = time.time()

    try:
        response = qa_chain.invoke({"query": request.question})
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        sources = []
        for doc in response['source_documents']:
            sources.append({
                "page": doc.metadata['page'],
                "content": doc.page_content[:200]
            })

        logger.info(f"Answer generated in {response_time}s")

        return AnswerResponse(
            question=request.question,
            answer=response['result'],
            sources=sources,
            response_time=response_time
        )

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating answer. Please try again."
        )