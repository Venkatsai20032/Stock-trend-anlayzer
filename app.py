import os
import pickle
import time
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJvNMuTbbe0FAixAckVtYkXXo7mb0vgao"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.title("Stock Trend Analyzer 📊")
st.sidebar.title("Stock News URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started ✅✅✅")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting... Started ✅✅✅")
    docs = text_splitter.split_documents(data)
    
    vector_index = FAISS.from_documents(docs, embeddings)
    pkl = vector_index.serialize_to_bytes()
    main_placeholder.text("Building Embeddings... ✅✅✅")
    time.sleep(2)
    
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

query = main_placeholder.text_input("Ask about stock trends: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
            vector_index = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl, allow_dangerous_deserialization=True)
            retriever = vector_index.as_retriever()
            
            response = retriever.get_relevant_documents(query)
            
            st.header("Answer")
            for doc in response:
                st.write(doc.page_content)
            
            st.subheader("Sources:")
            for doc in response:
                st.write(doc.metadata.get("source", "Unknown"))









