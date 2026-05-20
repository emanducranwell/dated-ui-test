import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_FILE = "greece-on-display-chunked.json"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("british_museum_objects")

def clean(value):
    if value is None:
        return ""
    return str(value).strip()

def make_chunk(record, chunk_type):
    if chunk_type == "core_facts":
        fields = [
            "Object type",
            "Museum number",
            "Title",
            "Culture",
            "Production date",
            "Production place",
            "Find spot",
            "Materials",
            "Technique",
            "Location"
        ]

    elif chunk_type == "description":
        fields = [
            "Description",
            "Subjects",
            "Assoc name",
            "Inscription",
            "Condition"
        ]

    elif chunk_type == "context":
        fields = [
            "Curators Comments",
            "Acq name (acq)",
            "Acq date",
            "Acq notes (acq)",
            "Acq notes (exc)",
            "Exhibition history",
            "Bib references"
        ]

    parts = []
    for field in fields:
        value = clean(record.get(field))
        if value:
            parts.append(f"{field}: {value}")

    return "\n".join(parts)

with open(DATA_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

documents = []
ids = []
metadatas = []

for index, chunk in enumerate(data[:500]):
    documents.append(chunk["text"])

    original_id = chunk.get("chunk_id", f"chunk_{index}")
    safe_id = f"{index}_{original_id}"

    ids.append(safe_id)
    metadatas.append(chunk["metadata"])

embeddings = model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Added {len(documents)} chunks to ChromaDB.")