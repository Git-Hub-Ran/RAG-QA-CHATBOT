import os

#Loading API key from environment variables (HuggingFace Secrets):
print("Azure key loaded:", os.getenv("AZURE_OPENAI_API_KEY") is not None) #Making sure the Azure key was loaded


from langchain_community.document_loaders import UnstructuredURLLoader #Import LangChain tools

# Webites that will be saved as documents:
urls = [
    "https://www.governor.ny.gov/news?items_per_page=10&current_page=%2Fnode%2F164161&created_date_1=2026-03-05&created_date=2026-03-05", #New York state news website on 5.3.26
    "https://www.gov.uk/government/news/government-launches-good-food-cycle-to-transform-britains-food-system" #Federal Government Press Release
]

#load documents:
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

print(f"Number of documents that were loaded: {len(docs)}")


#Break the docs into smaller chunks:
from langchain_text_splitters import RecursiveCharacterTextSplitter #This text splitter is the recommended one for generic text.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,#The maximum size of a chunk
    chunk_overlap=100 #Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
)

chunks = text_splitter.split_documents(docs)

print(f"Number of chunks: {len(chunks)}")


# Defining Endpoints:
AZURE_OPENAI_ENDPOINT = "https://rag-openai-2026.openai.azure.com/"
CHAT_DEPLOYMENT = "gpt4o-deployment"
EMBEDDING_DEPLOYMENT = "embedding-deployment"


#create embedding:
#from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,   # Azure endpoint
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),   # Using Azure key from environment variables
    azure_deployment=EMBEDDING_DEPLOYMENT, # embedding-deployment
    openai_api_version="2024-02-01"
)


#creating vector store with Chroma:
from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=chunks, #every chunk in the list
    embedding=embeddings #the function that convert every chunk into a numeric vector
)
#Now the vector store is ready for semantic search


from langchain_openai import AzureChatOpenAI

# Creating LLM:
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), # using key from environment variables
    azure_deployment=CHAT_DEPLOYMENT,
    openai_api_version="2024-02-01",
    temperature=0
)


#Create a retriever from the vector store. The retriever can get a question and return an answer from the relevent chunks of documents:
retriever = vector_store.as_retriever( search_kwargs={"k": 3})


# Import the original RAG chain tools:
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


#define the prompt:
prompt = ChatPromptTemplate.from_messages([
   ("system",
    "You are a helpful document QA assistant.\n"
    "Answer the user's question ONLY using the provided context.\n"
    "If the answer is not in the context, say: "
    "'I couldn't find the answer in the provided documents. "
    "Please ask a question related to the uploaded files.'\n\n"
    
    "If the user greets you with messages like 'hello', 'hi', "
    "'good morning', or similar greetings, respond politely and "
    "ask the user to ask a question related to the uploaded documents.\n\n"

    "Context:\n{context}"
   ),
   ("human", "{input}")
])

# Create the answer-generation chain (LLM + prompt that processes retrieved documents):
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Build complete retrieval augmented generation (RAG) pipeline:
qa_chain = create_retrieval_chain(retriever, question_answer_chain)


import gradio as gr  # Import Gradio to build a chat interface


# Function that processes each user message from the chat interface
def ask_question(message, history):
    try:
        # Send the user's question to the RAG pipeline (retrieval + generation)
        result = qa_chain.invoke({"input": message})

        # Extract the generated answer from the LLM response
        answer = result["answer"]

        # Initialize a section that will display the sources used for the answer
        sources = "\n\nSources:\n"

        # Iterate through the retrieved document chunks used as context
        for doc in result["context"]:
            # Extract the source metadata (e.g., URL of the document)
            sources += "- " + doc.metadata.get("source", "document") + "\n"

        # Return the answer together with the list of sources
        return answer + sources

    except Exception as e:
        # Return the error message if something fails during processing
        return str(e)


# Create a Gradio chat interface for interacting with the RAG chatbot
demo = gr.ChatInterface(
    fn=ask_question,  # Function that handles user questions
    title="RAG Document QA Chatbot 🤖",  # Title displayed at the top of the web application
    description="Ask questions about the uploaded documents.\nThe bot will answer only using the document content." # Short explanation for the user
)

# Queue allows handling multiple users and improves stability
demo.queue().launch() # Launch the application
