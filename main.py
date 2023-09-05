import os
import openai
import sys
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# openai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key  = "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"#os.getenv("OPENAI_API_KEY")

loader = [PyPDFLoader("rohit.pdf"),PyPDFLoader("shivam.pdf")]
docs = []

for load in loader:
  docs.extend(load.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings()
vectordb_fb = FAISS.from_documents(splits, embedding)

# vectordb_fb.similarity_search("what is python",k=3)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb_fb.as_retriever()
)

result = qa_chain({"query": str(input('inpt : '))})
print(result["result"])


