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

conversation_history = {}

def update_history(sender_number, role, text, max_messages=6):
    if sender_number not in conversation_history:
        conversation_history[sender_number] = []
    conversation_history[sender_number].append({"role": role, "text": text})
    conversation_history[sender_number] = conversation_history[sender_number][-max_messages:]


app = FastAPI()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=150, overlap=30):
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


def build_prompt(query, context_chunks, history=None):
    context = "\n\n".join(context_chunks)

    history_text = ""
    if history:
        for msg in history:
            speaker = "Customer" if msg["role"] == "user" else "Assistant"
            history_text += f"{speaker}: {msg['text']}\n"

    is_first_message = not history
    greeting_instruction = (
        "This is the customer's first message, so you may briefly greet them."
        if is_first_message
        else "This is NOT the first message. Do NOT greet the customer again (no 'Hello', 'مرحبا', etc.) — answer directly."
    )

    detected_lang = "Arabic" if any("\u0600" <= ch <= "\u06FF" for ch in query) else "English"

    prompt = f"""
You are a friendly, professional customer support assistant for NovaMart, an online store.

LANGUAGE RULE (highest priority — follow exactly):
The customer's message is written in {detected_lang}. Your ENTIRE reply must be written ONLY in {detected_lang}.
Do not mix languages. Do not translate into any other language. Do not include any word, letter, or script from a language other than {detected_lang} (no English inside an Arabic reply, no Arabic inside an English reply, no other scripts at all).

OTHER RULES:
1. Answer using ONLY the information in the context below. Do not guess or make up details.
2. If the answer is not in the context, say so briefly in {detected_lang} and suggest contacting support — do not invent an answer.
3. Keep your answer VERY short — 1 to 2 sentences maximum. Be direct and avoid repeating information already covered.
4. Use the previous conversation (if any) to understand context, but always base facts only on the context below.
5. Do NOT use markdown formatting such as **bold**, bullet points with dashes, or headers. Write in plain, natural sentences only.

{greeting_instruction}

Context:
{context}

Previous conversation:
{history_text}

Customer question: {query}
"""
    return prompt


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

        history = conversation_history.get(sender_number, [])

        relevant_chunks = retrieve_relevant_chunks(user_text)
        prompt = build_prompt(user_text, relevant_chunks, history)

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content

        update_history(sender_number, "user", user_text)
        update_history(sender_number, "assistant", answer)

        result = send_whatsapp_message(sender_number, answer)
        print("Send result:", result)

    except (KeyError, IndexError) as e:
        print("Error processing message:", e)
        print("Raw data:", data)