from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from groq import Groq

load_dotenv()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = "novamart_verify_123"

app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

text = extract_text_from_pdf("NovaMart_FAQ_Policies.pdf")
chunks = chunk_text(text)
embeddings = embedding_model.encode(chunks)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="novamart_knowledge")
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)


def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    return results["documents"][0]

def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful customer support assistant for NovaMart, an online store.
Answer the customer's question using ONLY the context below. If the answer is not
in the context, say you don't have that information and suggest contacting support.

Context:
{context}

Customer question: {query}

Answer clearly and concisely in a friendly, professional tone.
"""
    return prompt


def get_answer(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    prompt = build_prompt(query, relevant_chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    return {"error": "Verification failed"}

def send_whatsapp_message(to_number, message_text):
    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": message_text}
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        message = value["messages"][0]

        sender_number = message["from"]
        user_text = message["text"]["body"]

        answer = get_answer(user_text)
        send_whatsapp_message(sender_number, answer)

    except (KeyError, IndexError):
        pass

    return {"status": "received"}
