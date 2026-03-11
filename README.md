---
license: mit
sdk: gradio
emoji: 🏆
colorFrom: blue
colorTo: pink
---
# RAG QA Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on government news documents.

## Technologies

- LangChain
- Chroma Vector Database
- Azure OpenAI
- Gradio

## How It Works

1. Documents are loaded from public government news websites.
2. The text is split into smaller chunks.
3. Embeddings are created using Azure OpenAI.
4. The chunks are stored in a Chroma vector database.
5. When a user asks a question, relevant chunks are retrieved and used as context for the LLM.

## Deployment

The application is deployed using Hugging Face Spaces.

## Demo

Public URL: