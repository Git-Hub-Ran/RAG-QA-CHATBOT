# RAG QA Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions based on government news documents.  
The chatbot retrieves relevant text chunks from public websites and uses an LLM to generate accurate answers, citing the sources.

## Technologies

- **LangChain** – for building the RAG pipeline
- **Chroma Vector Database** – for storing document embeddings
- **Azure OpenAI** – for embeddings and chat generation
- **Gradio** – for creating an interactive web interface

## How It Works

1. **Load documents** from government news websites.
2. **Split text** into smaller chunks for better retrieval.
3. **Create embeddings** for each chunk using Azure OpenAI.
4. **Store chunks** in a Chroma vector database.
5. **Retrieve relevant chunks** when a user asks a question.
6. **Generate answers** using the LLM based on the retrieved context.
7. **Display sources** alongside the answer to ensure transparency.

## Project Structure
```
rag-document-qa/
│
├── app.py # Main RAG + Gradio app
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files to ignore in Git
```
## Deployment

The chatbot is deployed on **Hugging Face Spaces**.  
The live demo can be accessed publicly.

**Demo URL:** [RAG QA Chatbot](https://huggingface.co/spaces/DutchBrush/rag-qa-chatbot)

## Running Locally

1. Create a `.env` file with your Azure API key:
```AZURE_OPENAI_API_KEY=your_azure_key_here```

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
   python app.py
The Gradio interface will open locally and allow you to interact with the chatbot.

## Notes

- The documents are loaded at startup, which may take a few seconds.
- Answers are generated only from the retrieved context. If the information is not available, the bot will respond: "I don’t know."
- Ensure your Azure API key is valid and has access to the required OpenAI deployments.
