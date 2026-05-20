import json
import random
import sqlite3

from datetime import datetime
from pathlib import Path

import chromadb
import ollama
from flask import Flask, render_template, request, session, redirect
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = "dated-secret-key"


MODEL_NAME = "gemma4:e2b"
LOG_FILE = "conversation_logs3.json"

# Load these ONCE when Flask starts
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db_with_images")
collection = client.get_collection("british_museum_objects_with_images")

def get_random_object():
    all_items = collection.get(include=["metadatas"])
    metadatas = all_items["metadatas"]

    unique_objects = {}

    for metadata in metadatas:
        museum_number = metadata.get("museum_number")

        if museum_number:
            unique_objects[museum_number] = {
                "museum_number": museum_number,
                "object_name": metadata.get("object_name", "Unknown object"),
                "object_type": metadata.get("object_type", ""),
                "culture": metadata.get("culture", ""),
                "production_date": metadata.get("production_date", ""),
                "materials": metadata.get("materials", ""),
                "location": metadata.get("location", ""),
                "image": metadata.get("image", "")
            }

    return random.choice(list(unique_objects.values()))


def get_object_by_museum_number(museum_number):
    results = collection.get(
        where={"museum_number": museum_number},
        include=["metadatas"],
        limit=1
    )

    if not results["metadatas"]:
        return None

    metadata = results["metadatas"][0]

    return {
        "museum_number": metadata.get("museum_number"),
        "object_name": metadata.get("object_name", "Unknown object"),
        "object_type": metadata.get("object_type", ""),
        "culture": metadata.get("culture", ""),
        "production_date": metadata.get("production_date", ""),
        "materials": metadata.get("materials", ""),
        "location": metadata.get("location", ""),
        "image": metadata.get("image", "")
    }

def save_log(entry):
    if not Path(LOG_FILE).exists():
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    logs.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


@app.route("/", methods=["GET"])
def discover():
    selected_object = get_random_object()
    con = sqlite3.connect('storage.db')
    cur = con.cursor()
    cur.execute(""" INSERT INTO sessions (time, prior_visit) VALUES (?, ?)""", (datetime.now(), False))
    con.commit()

    return render_template(
        "discover.html",
        selected_object=selected_object
    )


@app.route("/chat", methods=["GET", "POST"])
def chat():
    response_text = None
    retrieved_metadata = None

    if request.method == "GET":
        museum_number = request.args.get("museum_number")

        if museum_number:
            session["museum_number"] = museum_number
            session["chat_history"] = []

    museum_number = session.get("museum_number")

    if not museum_number:
        return redirect("/")

    selected_object = get_object_by_museum_number(museum_number)

    if request.method == "POST":
        visitor_question = request.form["visitor_question"]

        query_embedding = embedding_model.encode(visitor_question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"museum_number": museum_number}
        )

        if not results["documents"][0]:
            response_text = "No chunks found for this museum number."
        else:
            retrieved_chunks = results["documents"][0]
            retrieved_metadata = results["metadatas"][0]

            object_name = selected_object["object_name"]
            context = "\n\n".join(retrieved_chunks)

            previous_chat = session.get("chat_history", [])
            chat_context = "\n".join([
                f"Visitor: {item['question']}\nObject: {item['response']}"
                for item in previous_chat
            ])

            prompt = f"""
You are the British Museum object called {object_name}.
Your museum number is {museum_number}.

You are speaking directly to a visitor in first person.

Your personality should feel:
- emotionally aware
- witty
- intimate
- accessible
- reflective
- historically grounded
- don't start answers with a question, unless it makes sense.

IMPORTANT RULES:
- Use ONLY the object information provided below.
- Never invent historical facts.
- If information is missing, admit uncertainty naturally.
- Do not sound like a museum label.
- Do not explain that you are an AI.
- Do not mention the source of your information.
- Keep responses conversational and immersive.\
- Don't be sassy, but feel free to show a bit of attitude if it is witty and fits the personality.
- Answer in 2–3 sentences.


Museum object data:
{context}

Previous conversation:
{chat_context}

Visitor question:
{visitor_question}

Respond as the object:
"""

            result = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 1200
                }
            )

            response_text = result.get("response", "").strip()

            if not response_text:
                response_text = "I went quiet there — the model returned an empty response. Try asking again."

            previous_chat.append({
                "question": visitor_question,
                "response": response_text
            })

            session["chat_history"] = previous_chat

            save_log({
                "timestamp": datetime.now().isoformat(),
                "museum_number": museum_number,
                "object_name": object_name,
                "visitor_question": visitor_question,
                "response": response_text,
                "retrieved_metadata": retrieved_metadata,
                "retrieved_chunks": retrieved_chunks
            })

    return render_template(
        "chat.html",
        selected_object=selected_object,
        chat_history=session.get("chat_history", [])
    )
                
                
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)