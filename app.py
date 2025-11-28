import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# GEMINI setup
gemini_key = os.getenv("GOOGLE_API_KEY")

# Instantiate Gemini client (per docs, e.g. https://ai.google.dev/gemini-api/docs/embeddings)
client = genai.Client(api_key=gemini_key)

gemini_ef = embedding_functions.GoogleGenaiEmbeddingFunction(
    model_name="gemini-embedding-001"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Generate embeddings for the document chunks using gemini client
def get_gemini_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    
    print("==== Generating embeddings... ====")
    return result.embeddings[0].values

for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Function to generate a response from Gemini
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)

    answer = response.text
    return answer

# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
