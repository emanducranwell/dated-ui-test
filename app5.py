# app5.py
from datetime import datetime
from flask import Flask, render_template, request, session, redirect

from config import CHROMA_DIR, MODEL_NAME, LOG_FILE, DB_PATH, SECRET_KEY
from services.rag import RAGService
from services.objects import ObjectsService
from services.logging import LoggingService
from services.sessions import SessionsService


app = Flask(__name__)
app.secret_key = SECRET_KEY

# Initialize services once at startup
rag_service = RAGService(chroma_dir=CHROMA_DIR, model_name=MODEL_NAME)
objects_service = ObjectsService(vectorstore=rag_service.vectorstore)
logging_service = LoggingService(log_file=LOG_FILE)
sessions_service = SessionsService(db_path=DB_PATH)


@app.route("/", methods=["GET"])
def discover():
    selected_object = objects_service.get_random()
    sessions_service.record_visit(prior_visit=False)
    return render_template("discover.html", selected_object=selected_object)


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        museum_number = request.args.get("museum_number")
        if museum_number:
            session["museum_number"] = museum_number
            session["chat_history"] = []

    museum_number = session.get("museum_number")
    if not museum_number:
        return redirect("/")

    selected_object = objects_service.get_by_museum_number(museum_number)

    if request.method == "POST":
        question = request.form["visitor_question"]
        result = rag_service.ask(
            question=question,
            museum_number=museum_number,
            object_name=selected_object["object_name"],
            chat_history=session.get("chat_history", []),
        )

        history = session.get("chat_history", [])
        history.append({"question": question, "response": result["answer"]})
        session["chat_history"] = history

        logging_service.save({
            "timestamp": datetime.now().isoformat(),
            "museum_number": museum_number,
            "object_name": selected_object["object_name"],
            "visitor_question": question,
            "response": result["answer"],
            "retrieved_chunks": result["chunks"],
        })

    return render_template(
        "chat.html",
        selected_object=selected_object,
        chat_history=session.get("chat_history", []),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)