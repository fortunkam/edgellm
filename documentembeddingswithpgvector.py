# pip install tiktoken faiss-cpu
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.pgvector import DistanceStrategy
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

host= os.environ['POSTGRES_HOST']
port= os.environ['POSTGRES_PORT']
user= os.environ['POSTGRES_USER']
password= os.environ['POSTGRES_PASSWORD']
dbname= os.environ['POSTGRES_DBNAME']

CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"

### Cloud
# embeddings = OpenAIEmbeddings()

### Edge
# embeddings = LlamaCppEmbeddings(model_path="./models/gpt4all-lora-quantized-new.bin")
# embeddings = LlamaCppEmbeddings(model_path="./models/ggml-vicuna-7b-4bit-rev1.bin", n_threads=16)
# embeddings = LlamaCppEmbeddings(model_path="./models/ggml-vicuna-13b-4bit-rev1.bin", n_threads=16)
embeddings = LlamaCppEmbeddings(model_path="./models/llama-2-7b.Q4_0.gguf", n_threads=4, n_gpu_layers=30)
# embeddings = LlamaCppEmbeddings(model_path="./models/falcon-7b-Q4_0-GGUF.gguf", n_threads=4, n_gpu_layers=30)



# Embedd your texts
#db = FAISS.from_documents(texts, embeddings)

COLLECTION_NAME = "blog_posts"
loaddocs = False

if loaddocs:
    # Load the document and split to fit in token context
    loader = TextLoader('data/satya-openai-announcement.txt')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} chunks")

    db = PGVector.from_documents(
        documents= texts,
        embedding = embeddings,
        collection_name= COLLECTION_NAME,
        distance_strategy = DistanceStrategy.COSINE,
        connection_string=CONNECTION_STRING)
else:
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

retriever = db.as_retriever()

# Retrieve relevant embeddings (Could also use a vector database here)
docs = retriever.get_relevant_documents("who is the CEO of OpenAI?")
for doc in docs:
    print("###")
    print(doc.page_content)