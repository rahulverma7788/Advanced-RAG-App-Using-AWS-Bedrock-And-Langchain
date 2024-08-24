import json
import os
import sys
import boto3
import streamlit as st
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Set page configuration
st.set_page_config(page_title="Chat PDF with AWS Bedrock", page_icon="💬", layout="wide")

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_Mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 512})
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.markdown("""
        <style>
        .main {
            background-color: #121212;
            color: #E0E0E0;
            padding: 2rem;
        }
        .stTextInput input, .stTextArea textarea {
            background-color: #1e1e1e;
            color: #E0E0E0;
            border: 1px solid #333333;
        }
        .stButton > button {
            background-color: #333333;
            border: 1px solid #444444;
            color: #E0E0E0;
        }
        .stButton > button:hover {
            background-color: #444444;
        }
        .stSidebar {
            background-color: #1e1e1e;
            color: #E0E0E0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Chat with PDF using AWS Bedrock 💬")
    st.markdown("<h2 style='text-align: center; color: #B3B3B3;'>Ask questions about your PDF documents</h2>", unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Ask a Question")
            user_question = st.text_input("Type your question here...")

        with col2:
            st.sidebar.header("Update or Create Vector Store")
            st.sidebar.write("Click the button below to update the FAISS vector store with the latest PDF documents.")

            if st.sidebar.button("Update Vectors"):
                with st.spinner("Processing..."):
                    try:
                        docs = data_ingestion()
                        get_vector_store(docs)
                        st.sidebar.success("Vector store updated successfully!")
                    except Exception as e:
                        st.sidebar.error(f"An error occurred: {e}")

    if st.button("Get Mistral Output"):
        with st.spinner("Generating response with Mistral..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_Mistral_llm()
                answer = get_response_llm(llm, faiss_index, user_question)
                st.subheader("Mistral's Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.button("Get Llama3 Output"):
        with st.spinner("Generating response with Llama3..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama3_llm()
                answer = get_response_llm(llm, faiss_index, user_question)
                st.subheader("Llama3's Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("<footer style='text-align: center; color: #B3B3B3;'>Powered by AWS Bedrock & Streamlit</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
