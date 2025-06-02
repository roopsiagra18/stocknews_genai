import langchain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
file_path="vector_index.pkl"

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.9)
# Instantiate Gemini LLM

url_counts = 3

urls = []
for i in range(url_counts):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    loaders = UnstructuredURLLoader(urls =urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index_gemini = FAISS.from_documents(docs,embedding=embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(20)
    with open(file_path, "wb") as f:
        pickle.dump(vector_index_gemini, f)
    main_placeholder.text("Storing Vector to pickle file...✅✅✅")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7  # sometimes helps with prompt formatting
)
print(llm)
print(type(llm)) 
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)
            main_placeholder.text("Building LLM Chain...✅✅✅")
            retriever = vectorIndex.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)
            print(result)
            st.header("Answer")
            st.write(result["answer"])
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
