import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Modern 2026 LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# 1. Load the secret key from your .env file
load_dotenv()

# 2. Setup the AI "Brain" & "Energy Reader"
# 2. Setup the AI "Brain" & "Energy Reader"
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model_name="gpt-4o-mini", 
    temperature=0.2
)
embeddings = OpenAIEmbeddings()

# 3. Build the Web Interface
st.title("🔮 Mystical RAG: Ask Your PDF")
st.write("Upload a document and let the AI read it for you!")

# The File Uploader Button
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    raw_text = ""
    
    # --- SAFETY NET (Error Handling) ---
    try:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    except Exception as e:
        st.error(f"Uh oh! The app couldn't read this PDF. Error: {str(e)}")
        
    # Check if PDF is empty or just images
    if not raw_text.strip():
        st.error("This PDF seems to be empty or consists only of images. Please upload a text-based PDF.")
    else:
        st.success(f"Successfully extracted {len(raw_text)} characters!")
        
        # --- THE RAG PIPELINE ---
        
        # A. The Shuffler (Text Splitter)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)
        
        # B. The Velvet Table (FAISS Vector Store)
        with st.spinner("Building the knowledge base... 🔮"):
            doc_search = FAISS.from_texts(texts, embeddings)
            
        st.success("Cards are on the table! Ask your question below.")
        
        # C. The User Question Input
        user_question = st.text_input("What would you like to know from this document?")
        
        if user_question:
            # D. The Guide & Reader (RetrievalQA)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=doc_search.as_retriever()
            )
            
            with st.spinner("Consulting the document... 🕵️‍♂️"):
                # E. Get the Answer! 
                response = chain.invoke({"query": user_question})
                
                st.write("### 🔮 The Answer:")
                st.info(response["result"])