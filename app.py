

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA

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

model_name = "microsoft/Phi-3.5-mini-instruct"

# function for loading 4-bit quantized model
## If you will have gpu then used this method for load huggingface model

# def load_quantized_model(model_name: str):
#     """
#     model_name: Name or path of the model to be loaded.
#     return: Loaded quantized model.
#     """
#     bnb_config = BitsAndBytesConfig(
# #load_in_4bit=True,
#     load_in_4bit=False,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#     model_name, quantization_config=bnb_config, trust_remote_code=True, device_map='mps')
#     return model


# initializing tokenizer
def initialize_tokenizer(model_name: str):
    """
    model_name: Name or path of the model for tokenizer initialization.
    return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer


tokenizer = initialize_tokenizer(model_name)

# model = load_quantized_model(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name,
    device_map="mps", 
    torch_dtype="auto", 
    trust_remote_code=True)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline)

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
