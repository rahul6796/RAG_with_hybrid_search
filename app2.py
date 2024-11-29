

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
     

from dotenv import load_dotenv
import os
load_dotenv()

hug_api_key = os.getenv('HUGF_KEYS')


doc_path = 'data/2005.11401v4.pdf'

# load documnets
loader=PyPDFLoader(doc_path)
docs=loader.load()


# Split text into chunk:
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=30)
chunks = splitter.split_documents(docs)

# Load Embedding Model and store data into vector-database:

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hug_api_key, model_name="BAAI/bge-base-en-v1.5")
vectorstore=Chroma.from_documents(chunks,embeddings)

# Define The Vector-Retriever
vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
# print(vectorstore_retreiver)

# Define The Key-Word-Retriever
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 3

# print(keyword_retriever)


# Define The Ensemble-Retriever
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])

# Mixing vector search and keyword search for Hybrid search
# hybrid_score = (1 — alpha) * sparse_score + alpha * dense_score

llm = OllamaLLM(model="mistral")

# Define normal Chain:
normal_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore_retreiver
)

# Define hybrid chain:
hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ensemble_retriever
)

response1 = normal_chain.invoke("What is Abstractive Question Answering?")
print(response1.get("result"))

print('++++++++++++++++ HYBRIDE SEARCH +++++++++++++')


response2 = hybrid_chain.invoke("What is Abstractive Question Answering?")
print(response2.get("result"))
