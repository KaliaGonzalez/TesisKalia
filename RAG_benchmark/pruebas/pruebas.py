# Todas las librerias y frameworks que vamos a necesitar
import ollama
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
import re

# Configurar allow_reset en True
settings = Settings(allow_reset=True)

# Crear el cliente persistente de Chroma
client = chromadb.PersistentClient(path="../chromaDataBase", settings=settings)
client.heartbeat()
client.reset()  # - empties and completely resets the database. This is destructive and not reversible.

# Creacion de la coleccion de prueba
try:
    collection = client.create_collection(name="pruebas")
except:
    collection = client.get_collection(name="pruebas")

# Cargar documento y volverlo chunks
file_path = "C:/Users/nicod/OneDrive/Escritorio/Tesis/Referencias/preg_frecuentes_fac.pdf"

loader = PyPDFLoader(file_path)
document = loader.load()
filtered_documents = [doc for i, doc in enumerate(document) if i not in [3, 4]]
# Remove commas and special characters from filtered documents
with open("filtered_documents2.txt", "w", encoding="utf-8") as f:
    for doc in filtered_documents:
        f.write(doc.page_content + "\n")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(filtered_documents)
print(len(chunked_documents))

# Cargar los datos vectorizados a la base de datos de Chroma
try:
    for i, d in enumerate(chunked_documents):
        response = ollama.embed(model="jina/jina-embeddings-v2-base-es", input=d.page_content)
        embeddings = response["embeddings"]
        collection.add(
        ids=[str(i)],
        embeddings=embeddings,
        documents=[d.page_content]
    )
    print(f"Added {len(chunked_documents)} chunks to chroma db's collection: pruebas")
except Exception as e:
    print(f"An error occurred: {e}")
