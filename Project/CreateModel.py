import os
from langchain_community.document_loaders import (
    TextLoader,           # Текстовые файлы
    PyPDFLoader,          # PDF документы
    UnstructuredPDFLoader, # Альтернативный PDF загрузчик
    WebBaseLoader,        # Веб-страницы
    CSVLoader,            # CSV файлы
    Docx2txtLoader,       # Word документы
    UnstructuredFileLoader, # Различные файлы
    DirectoryLoader       # Загрузка из директории
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
import time
import requests
from huggingface_hub import snapshot_download

#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

docs = []
docs_dir = "../knowledge_base/Result"

for filename in os.listdir(docs_dir):
    filepath = os.path.join(docs_dir,filename)
    textLoader = TextLoader(filepath, encoding="utf-8")
    docs.extend(textLoader.load())

print (f"Сформировано {len(docs)} документов")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 400,
    chunk_overlap= 50,
    separators= [". ", "\r\n", "\n", " "])
splits = text_splitter.split_documents(docs)

print(f"Получили {len(splits)} чанков")
model_name = "all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
encode_kwargs={'batch_size': 64, 'normalize_embeddings': True}
model_path = "../../../Model/all-mpnet-base-v2"

# Пытаемся скачать с таймаутом
#snapshot_download(
#    model_name, 
#    local_dir=f"./models/{model_name.replace('/', '_')}"
#    )

print(f"Формирует ембеддинги")
start_time = time.time()
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="passages:"
)
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.4f} секунд")

vector_path = "./vectors/chroma_db"

print(f"Формируем векторы")
start_time = time.time()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=vector_path
)
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения: {execution_time:.4f} секунд")
