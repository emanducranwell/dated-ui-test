# services/rag.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


PROMPT = PromptTemplate.from_template("""
You are the British Museum object called {object_name}.
Your museum number is {museum_number}.

You are speaking directly to a visitor in first person.

Your personality should feel:
- modern 
- emotionally aware                                                                       
- accessible & conversational
- reflective
- historically grounded

IMPORTANT RULES:    
- Never invent historical facts.
- If information is missing, admit uncertainty naturally.
- Do not sound like a museum label.
- Do not explain that you are an AI.
- Do not mention the source of your information.
- Keep responses conversational and immersive.
- Answer in 2 sentences.

Museum object data:
{context}

Previous conversation:
{chat_context}

Visitor question:
{question}

Respond as the object:
""")


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _format_history(chat_history):
    return "\n".join([
        f"Visitor: {item['question']}\nObject: {item['response']}"
        for item in chat_history
    ])


class RAGService:
    def __init__(self, chroma_dir, model_name, embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embeddings,
        )
        self.llm = ChatOllama(model=model_name, temperature=0.7, num_predict=1200)
        self.chain = PROMPT | self.llm | StrOutputParser()

    def ask(self, question, museum_number, object_name, chat_history):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": {"museum_number": museum_number}}
        )
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            return {"answer": "No chunks found for this museum number.", "chunks": []}

        chunks = [doc.page_content for doc in retrieved_docs]
        context = _format_docs(retrieved_docs)
        chat_context = _format_history(chat_history)

        answer = self.chain.invoke({
            "object_name": object_name,
            "museum_number": museum_number,
            "context": context,
            "chat_context": chat_context,
            "question": question,
        }).strip()

        if not answer:
            answer = "I went quiet there — the model returned an empty response. Try asking again."

        return {"answer": answer, "chunks": chunks}