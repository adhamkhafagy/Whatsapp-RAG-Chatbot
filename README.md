# NovaMart WhatsApp Support Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions over WhatsApp, powered by Hugging Face embeddings, ChromaDB, BM25 hybrid search, cross-encoder re-ranking, and Groq, connected through the Meta WhatsApp Cloud API.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## Table of Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview
This project is a customer support chatbot for a fictional online store, NovaMart, that answers questions about shipping, returns, payments, warranty, and more directly over WhatsApp. It uses a full retrieval pipeline — combining semantic and keyword search, re-ranking, and context compression — to ground every answer in the company's policy document rather than the LLM's general knowledge, while also remembering conversation context and replying in the customer's language.

## Pipeline
```
PDF → Smart Chunking → Embeddings → ChromaDB
    → Hybrid Retrieval → Re-ranking → Context Compression → LLM
```

## How It Works
1. The company policy PDF is parsed and split using **smart chunking**, which splits on section boundaries (titles/headings) instead of arbitrary word counts, keeping each policy topic intact.
2. Each chunk is embedded using a Hugging Face sentence-transformers model and stored in **ChromaDB**.
3. A **BM25 keyword index** is built alongside the vector store for exact-term matching.
4. When a customer messages the bot, their question is run through **hybrid retrieval** — combining semantic similarity search (ChromaDB) with keyword search (BM25) — to catch both conceptual and literal matches.
5. The combined candidates are **re-ranked** using a cross-encoder model for higher-precision relevance scoring.
6. The top results are passed through **context compression**, where an LLM call extracts only the sentences directly relevant to the question, reducing noise before the final prompt.
7. The compressed context, conversation history, and question are sent to Groq's LLM, which generates a short, grounded, friendly answer — automatically replying in Arabic or English depending on the customer's language.
8. The answer is sent back to the customer via the WhatsApp Cloud API.

## Features
- Real conversational chatbot accessible directly through WhatsApp
- Hybrid retrieval (semantic + keyword) for more reliable matches
- Cross-encoder re-ranking for higher-precision context selection
- Context compression step to reduce noise and token usage before generation
- Per-user conversation memory (last 6 messages) for natural follow-up questions
- Automatic language detection — replies in Arabic or English to match the customer
- Concise, markdown-free responses tuned for WhatsApp readability
- Answers are grounded strictly in the provided document, reducing hallucination
- FastAPI webhook architecture, deployed permanently on Railway

## Tech Stack
- Python
- FastAPI
- Sentence-Transformers (Hugging Face) — embeddings + cross-encoder re-ranker
- ChromaDB — vector store
- rank-bm25 — keyword search
- Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- Meta WhatsApp Cloud API
- Railway — permanent hosting

## Project Structure
```
whatsapp-rag-chatbot/
├── whatsapp_app.py
├── NovaMart_FAQ_Policies.pdf
├── requirements.txt
├── .gitignore
├── .env (not tracked)
└── README.md
```

## Setup
1. Clone the repository
```bash
git clone https://github.com/adhamkhafagy/Whatsapp-RAG-Chatbot.git
cd Whatsapp-RAG-Chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables
```
GROQ_API_KEY=your_groq_api_key
WHATSAPP_ACCESS_TOKEN=your_whatsapp_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
```

4. Run the FastAPI server
```bash
uvicorn whatsapp_app:app --reload
```

5. For permanent deployment, connect the repository to Railway (or a similar host), set the same environment variables there, and expose the service publicly.

6. Configure the webhook in the Meta App Dashboard (WhatsApp > Configuration) with your public URL + `/webhook`, and subscribe to the `messages` field.

7. Link the app to the WhatsApp Business Account using a permanent System User token:
```bash
curl.exe -X POST "https://graph.facebook.com/v20.0/{WABA_ID}/subscribed_apps" -H "Authorization: Bearer {SYSTEM_USER_TOKEN}"
```

## Usage
Send a message to the connected WhatsApp number asking about shipping, returns, payment methods, warranty, or account details — in Arabic or English — and the bot will respond with a short, accurate answer grounded in NovaMart's policy document, remembering the context of the conversation as it continues.

## Future Improvements
- Support multilingual documents beyond Arabic/English
- Allow the knowledge base to be updated without restarting the server
- Add analytics on common customer questions
- Open the WhatsApp number to any user via Meta Business Verification

## License
This project is licensed under the MIT License.
