# NovaMart WhatsApp Support Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions over WhatsApp, powered by Hugging Face embeddings, ChromaDB, Groq, and the Meta WhatsApp Cloud API.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20API-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview
This project is a customer support chatbot for a fictional online store, NovaMart, that answers questions about shipping, returns, payments, warranty, and more directly over WhatsApp. Instead of relying on the LLM's general knowledge, the bot retrieves the most relevant information from a company FAQ/policy document using semantic search, then generates an accurate, grounded answer.

## How It Works
1. A company policy PDF is parsed and split into overlapping text chunks.
2. Each chunk is converted into a vector embedding using a Hugging Face sentence-transformers model.
3. Embeddings are stored in a local ChromaDB vector database.
4. When a customer messages the bot on WhatsApp, their question is embedded and matched against the stored chunks using semantic similarity search.
5. The most relevant chunks are passed as context to Groq's LLM, which generates a grounded answer.
6. The answer is sent back to the customer via the WhatsApp Cloud API.

## Features
- Real conversational chatbot accessible directly through WhatsApp
- Semantic search over a company knowledge base (no keyword matching required)
- Answers are grounded strictly in the provided document, reducing hallucination
- FastAPI webhook architecture, deployable to any cloud server
- Fast, free-tier-friendly embedding model (`all-MiniLM-L6-v2`)

## Tech Stack
- Python
- FastAPI
- Sentence-Transformers (Hugging Face)
- ChromaDB
- Groq API (`llama-3.3-70b-versatile`)
- Meta WhatsApp Cloud API
- ngrok (for local development tunneling)

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
git clone https://github.com/adhamkhafagy/whatsapp-rag-chatbot.git
cd whatsapp-rag-chatbot
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

5. Expose the local server with ngrok (for development/testing)
```bash
ngrok http 8000
```

6. Configure the webhook in the Meta App Dashboard (WhatsApp > Configuration) with the ngrok URL + `/webhook`, and subscribe to the `messages` field.

7. Link the app to the WhatsApp Business Account:
```bash
curl.exe -X POST "https://graph.facebook.com/v20.0/{WABA_ID}/subscribed_apps" -H "Authorization: Bearer {ACCESS_TOKEN}"
```

## Usage
Send a message to the connected WhatsApp test number asking about shipping, returns, payment methods, warranty, or account details, and the bot will respond with an answer grounded in NovaMart's policy document.

## Future Improvements
- Deploy the FastAPI server to a persistent cloud host (Render, Railway, or a VPS) instead of ngrok
- Support multi-turn conversation memory
- Add multilingual support (Arabic/English)
- Allow the knowledge base to be updated without restarting the server

## License
This project is licensed under the MIT License.
