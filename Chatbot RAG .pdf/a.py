import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

class Chatbot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")  # Ambil dari environment variable
        if not self.api_key:
            raise ValueError("API Key is missing. Set GROQ_API_KEY in the .env file.")
        
        self.llm = ChatGroq(groq_api_key=self.api_key, model_name="Llama3-8b-8192")
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert chatbot in football. Answer user inquiries based on the provided context.
            Context:
            {context}
            
            User: {user_input}
            Assistant:
            """
        )
        
        # Load and process football-related documents
        self.loader = PyPDFDirectoryLoader("./stat")  # Folder with football-related PDFs
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.final_documents = self.text_splitter.split_documents(self.docs[:20])
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = FAISS.from_documents(self.final_documents, self.embeddings)
        self.retriever = self.vector_db.as_retriever()
        
        self.chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
    
    def get_response(self, user_input):
        response = self.chain.run(user_input)
        return response

chatbot = Chatbot()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = chatbot.get_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
