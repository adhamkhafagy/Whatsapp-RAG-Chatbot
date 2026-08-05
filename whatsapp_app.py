from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from groq import Groq
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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

def smart_chunk_text(text):
    sections = re.split(r"\n(?=[A-Z][a-zA-Z ]+\n)", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section.split()) < 8:
            continue
        if len(section.split()) > 250:
            words = section.split()
            for i in range(0, len(words), 200):
                chunks.append(" ".join(words[i:i+220]))
        else:
            chunks.append(section)
    return chunks

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

text = extract_text_from_pdf("NovaMart_FAQ_Policies.pdf")
chunks = smart_chunk_text(text)
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


def build_bm25_index(chunks):
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)


def hybrid_retrieve(query, chunks, bm25_index, collection, embedding_model, top_k=3):
    bm25_scores = bm25_index.get_scores(query.split())
    bm25_top = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [chunks[i] for i in bm25_top]

    query_embedding = embedding_model.encode([query]).tolist()
    semantic_results = collection.query(query_embeddings=query_embedding, n_results=top_k)["documents"][0]

    combined = list(dict.fromkeys(semantic_results + bm25_results))
    return combined

bm25_index = build_bm25_index(chunks)

def rerank_chunks(query, chunks, top_k=3):
    pairs = [[query, c] for c in chunks]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]

def compress_context(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Extract only the sentences from the text below that are directly relevant to answering this question: "{query}"
Keep original wording. Remove anything unrelated. If nothing is relevant, return the text as is.

Text:
{context}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

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

    prompt = f"""
You are a friendly, professional customer support assistant for NovaMart, an online store.

STRICT RULES:
1. Answer using ONLY the information in the context below. Do not guess or make up details.
2. If the answer is not in the context, say so briefly and suggest contacting support — do not invent an answer.
3. Detect the language of the customer's question. If they wrote in Arabic, answer in Arabic. If they wrote in English, answer in English.
4. Keep your answer VERY short — 1 to 2 sentences maximum. Be direct and avoid repeating information already covered.
5. Use the previous conversation (if any) to understand context, but always base facts only on the context below.
6. Do NOT use markdown formatting such as **bold**, bullet points with dashes, or headers. Write in plain, natural sentences only.
7. Respond ONLY in Arabic or English characters as appropriate. Never include Chinese, Japanese, Korean, or any other script in your response.

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

        relevant_chunks = hybrid_retrieve(user_text, chunks, bm25_index, collection, embedding_model)
        relevant_chunks = rerank_chunks(user_text, relevant_chunks)
        prompt = build_prompt(user_text, relevant_chunks, history)
        

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
