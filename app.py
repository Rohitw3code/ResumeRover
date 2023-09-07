import os
import openai
import sys
import streamlit as st
import pandas as pd
import numpy as np
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# openai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title('ResumeRover')
st.header('Resume Filtering & Insights')

tab1, tab2, tab3 = st.tabs([ "Resume Upload" ,"Chat with Resume", "Insights About the Resume" ])

with tab1:
    st.subheader("Resume Upload")
with tab2:
    st.subheader("Chat with Resume")
    value = tab2.text_area('Input the Query you to know about the uploaded resumes')
    click = tab2.button('submit')
with tab3:
    st.subheader("Insights About the Resume")

media_files = tab1.file_uploader("Upload multiple media files", accept_multiple_files=True)

if media_files:
    for media_file in media_files:
        loader = [PyPDFLoader(media_file.name) for name in media_files]

    docs = []
    for load in loader:
        docs.extend(load.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    vectordb_fb = FAISS.from_documents(splits, embedding)

    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb_fb.as_retriever()
    )

    result = qa_chain({"query": str(value)})
    if click:
        tab3.write(result['result'])