import weaviate

from dotenv import load_dotenv

from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )
from langchain import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate



from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM

import os
load_dotenv()

hug_api_key = os.getenv('HUGF_KEYS')
weviate_url = os.getenv('WEVIATE_CLUSTER')
weviate_api_key = os.getenv('WEVIATE_API_KEY')


WEAVIATE_URL = weviate_url
WEAVIATE_API_KEY = weviate_api_key


client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={
         "X-HuggingFace-Api-Key": hug_api_key
    },
)
    
# print(client.is_ready())

# print(client.schema.get())

# Define the schema for weviate.
schema = {
    "classes": [
        {
            "class": "RAG",
            "description": "Documents for RAG",
            "vectorizer": "text2vec-huggingface",
            "moduleConfig": {"text2vec-huggingface": {"model": "sentence-transformers/all-MiniLM-L6-v2", "type": "text"}},
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-huggingface": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}


# client.schema.create(schema)
# print(client.get_meta())

## Define the langchain-weaviate_hybride search retriever
retriever = WeaviateHybridSearchRetriever(
    alpha = 0.5,               # defaults to 0.5, which is equal weighting between keyword and semantic search
    client = client,           # keyword arguments to pass to the Weaviate client
    index_name = "RAG",  # The name of the index to use
    text_key = "content",         # The name of the text key to use
    attributes = [], # The attributes to return in the results
    create_schema_if_missing=True,
)


# define data:
doc_path = 'data/2005.11401v4.pdf'

# load documnets
loader=PyPDFLoader(doc_path)
docs=loader.load()

# Adding documents to the Weaviate:
retriever.add_documents(docs)

# print(retriever.invoke("what is RAG token?")[0].page_content)
# print(retriever.invoke('What is RAG token?', score = True))


# Load MOdel LLM:
llm = OllamaLLM(model="mistral")



# Define system prompt:

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
   

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{query}"),
    ]
)
    
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you do not have the relevant information needed to provide a verified answer, don't try to make up an answer.
When providing an answer, aim for clarity and precision. Position yourself as a knowledgeable authority on the topic, but also be mindful to explain the information in a manner that is accessible and comprehensible to those without a technical background.
Always say "Do you have any more questions pertaining to this instrument?" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""



prompt = PromptTemplate.from_template(template)

# Define QA chain:
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Define Hybride chain: 
hybrid_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,)

result1 = hybrid_chain.invoke("what is natural language processing?")
# print(result1)

# print(result1['result'])


## ReRank Technique by using Cohere API:

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

cohere_api_key = os.getenv('COHERE_API_KEY')

compressor = CohereRerank(cohere_api_key=cohere_api_key)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )

# compressed_docs = compression_retriever.get_relevant_documents('What is RAG token?')
# # Print the relevant documents from using the embeddings and reranker
# print(compressed_docs[0])


hybrid_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=compression_retriever
)

response = hybrid_chain.invoke("What is Abstractive Question Answering?")    
print(response.get("result"))
     