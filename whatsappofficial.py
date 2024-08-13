from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from uuid import uuid4
import sqlite3

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Global variables
docs = []
retriever = None
data_fetched = False
chat_histories = {}  # Dictionary to store chat history per user
processed_messages = set()

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# WhatsApp Graph API credentials
WHATSAPP_API_URL = "https://graph.facebook.com/v20.0/402608122930308/messages"
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")

# Function to fetch content from a URL
def fetch_content_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # List of tags to extract text from
        tags_to_extract = ['p', 'h1', 'h2', 'h3', 'li', 'span', 'div', 'label']
        
        # Extract text from all specified tags
        content_text = ' '.join([element.get_text(strip=True) for tag in tags_to_extract for element in soup.find_all(tag)])
        
        # Extract social media links
        social_media_links = []
        social_media_classes = [
            'elementor-icon elementor-social-icon elementor-social-icon-facebook-f elementor-repeater-item-7e47d2d',
            'elementor-icon elementor-social-icon elementor-social-icon-youtube elementor-repeater-item-e5230fa',
            'elementor-icon elementor-social-icon elementor-social-icon-linkedin elementor-repeater-item-e22c365',
            'elementor-icon elementor-social-icon elementor-social-icon-instagram elementor-repeater-item-72bbaff'
            # Add other social media classes as needed
        ]
        
        for cls in social_media_classes:
            links = [a.get('href') for a in soup.find_all('a', class_=cls, href=True)]
            social_media_links.extend(links)
        
        # Append social media links to content_text
        if social_media_links:
            content_text += ' ' + ' '.join(social_media_links)
        
        return {
            "text": content_text,
            "links": social_media_links
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# List of URLs to scrape content from
urls = [
    "https://brbgroup.org",
    "https://www.brbgroup.org/who-we-are/",
    "https://www.brbgroup.org/business-unit/",
    "https://www.brbgroup.org/why-brb/",
    "https://www.brbgroup.org/esg-sustainability/",
    "https://www.brbgroup.org/communities/",
    "https://www.brbgroup.org/media/#media-news",
    "https://www.brbgroup.org/finding-your-oasis-the-growing-demand-for-suburban-properties-in-karachi/",
    "https://www.oasispark.com.pk/",
    "https://brbgroup.org/construction-and-real-estate-solutions/",
    "https://brbgroup.org/brb-technologoies/",
    "https://brbgroup.org/brb-marketing-pvt-ltd/",
    "https://brbgroup.org/brb-engineering-pvt-ltd/",
    "https://brbgroup.org/brb-urbanscape-pvt-ltd/",
    "https://brbgroup.org/brb-trading-pvt-ltd/",
    "https://brbgroup.org/brb-foundation/",

    # Add more URLs as needed
]

# Fetch and store data
async def fetch_and_store_data():
    global docs, retriever, data_fetched
    docs = []  # Reset the docs list
    for url in urls:
        result = fetch_content_from_url(url)
        if result:
            content_text = result["text"]
            # Create a document for text content
            doc = Document(page_content=content_text, metadata={"url": url})
            docs.append(doc)
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=550, chunk_overlap=50
    )
    chunked_documents = text_splitter.split_documents(docs)

    # Generate embeddings for the chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
    )

    # Create the retriever
    retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 chunks
    data_fetched = True

# Define RAG prompt
RAG_PROMPT = """\
You are BRB Group's AI assistant on WhatsApp. Your role is to provide helpful, concise, and friendly responses to customer queries about BRB Group's services, properties, and initiatives. Use the following context to answer the user's question. If the information isn't in the context, then apologize and offer to assist with something else related to BRB Group.

Chat History:
{chat_history}

Customer Question: {question}

Relevant Information: {context}

Please respond in a friendly, professional manner, keeping your answer brief (preferably within 2-3 sentences) to suit WhatsApp messaging. If appropriate, end your response by asking if the customer needs any further information.

Additionally, don't respond to anything except questions about BRB. For example, if anyone asks any mathematics questions or science questions, general knowledge questions, etc., then don't answer it, and apologize and tell them you provide information about BRB Group's services, properties, and initiatives.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

# Define retrieval-augmented generation chain
def create_retrieval_augmented_generation_chain():
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
        | RunnablePassthrough.assign(context=itemgetter("context"), chat_history=itemgetter("chat_history"))
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context"), "chat_history": itemgetter("chat_history")}
    )

# Function to update chat history
def update_chat_history(user, question, response):
    if user not in chat_histories:
        chat_histories[user] = []  # Initialize history for new users
    chat_histories[user].append(f"Q: {question}\nA: {response}")

def store_chat_in_db(user, message, response):
    # Connect to the database
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    
    # Insert the chat record
    cursor.execute('''
    INSERT INTO chat_history (user, message, response)
    VALUES (?, ?, ?)
    ''', (user, message, response))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

@app.post("/fetch-data")
async def fetch_data():
    try:
        await fetch_and_store_data()
        return PlainTextResponse("Data fetched and stored successfully.", status_code=200)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return PlainTextResponse("Failed to fetch data", status_code=500)

@app.post("/webhook")
async def webhook(request: Request):
    global retriever, data_fetched, processed_messages

    # Use JSON data to retrieve 'from' and 'body'
    payload = await request.json()
    entry = payload.get('entry', [])[0]
    changes = entry.get('changes', [])[0]
    messages = changes.get('value', {}).get('messages', [])

    for message in messages:
        from_number = message.get('from')
        message_body = message.get('text', {}).get('body')
        message_id = message.get('id')

        # Check if the message has already been processed
        if message_id in processed_messages:
            continue
        
        # Log the incoming message
        print(f"Received message from {from_number}: {message_body}")

        if not from_number:
            print("Failed to retrieve 'From' number from request.")
            return PlainTextResponse("Missing 'From' number", status_code=400)

        if not data_fetched:
            await fetch_and_store_data()

        # Create and return a response message
        try:
            if retriever:
                # Handle queries with the chatbot
                retrieval_augmented_generation_chain = create_retrieval_augmented_generation_chain()
                response = await retrieval_augmented_generation_chain.ainvoke({
                    "question": message_body, 
                    "chat_history": chat_histories.get(from_number, [])
                })

                print(response)
                response_message = response['response'].content

                # Update chat history with the latest question and response
                update_chat_history(from_number, message_body, response_message)

                # Store chat in the database
                store_chat_in_db(from_number, message_body, response_message)
            else:
                response_message = "Data not initialized. Please fetch data first."

            # Send a response message using the WhatsApp Graph API
            send_whatsapp_message(from_number, response_message)

            # Mark the message as processed
            processed_messages.add(message_id)

        except Exception as e:
            print(f"Failed to send message: {e}")
            return PlainTextResponse("Failed to send message", status_code=500)

    return PlainTextResponse("OK")

# Function to send a message using WhatsApp Graph API
def send_whatsapp_message(to_number: str, text: str):
    url = WHATSAPP_API_URL
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json',
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {
            "body": text
        }
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Sent message response: {response.json()}")
