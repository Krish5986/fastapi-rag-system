from fastapi import FastAPI,UploadFile,File,BackgroundTasks
from fastapi import Request # Add Request to your existing fastapi imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import shutil
import os
import fitz 
import faiss
import numpy as np 
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import time

class QueryRequest(BaseModel):
    question: str


app = FastAPI(title='RAG Question Answering API')

#Initialize Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#create a directory to temporarily store uploaded files 
os.makedirs("uploaded_docs",exist_ok=True)

#Initialize Embedding Model & FAISS
print("Loading embedding model...")
#using a fast lightweight model 
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize a local FASS index
dimension = 384 #Output dimension for the MiniLM model
faiss_index = faiss.IndexFlatL2(dimension)

#FAISS only stores numbers. We need a list to store the actual text chunks 
#So we can return them when a user asks a question!
document_store = []

#Initialize LLM 
print("Configuring LLM...")
load_dotenv() #reads .env file

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in .env file!")
else:
    genai.configure(api_key=api_key)

#Using 2.5 flash model 
llm = genai.GenerativeModel('gemini-2.5-flash')


#Helper Functions
def extract_text(file_path:str,filename:str) -> str:
    """Extracts text from PDF or TXT files."""
    text = ""
    if filename.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    elif filename.endswith(".txt"):
        with open(file_path,"r",encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    return text

def chunk_text(text:str, chunk_size:int = 300,overlap: int= 50) -> list[str]:
    """
    Splits text into chunks of roughly 'chunk_size' words,
    with an overlap of 'overlap' words to maintain context.
    """
    words = text.split()
    chunks = []
    for i in range(0,len(words),chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks

#Background task
def process_document(file_path:str,filename: str):
    """
    This function runs in the background.
    We will add the PDF/TXT parsing,chunking and embedding logic here later.
    """
    print(f"Background processing started for: {filename}")

    try:
        #1. Extract and Chunk
        raw_text = extract_text(file_path,filename)
        chunks = chunk_text(raw_text,chunk_size=300,overlap=50)
        print(f"Document split into {len(chunks)} chunks.")

        #2. Generate embeddings
        print("Generating embeddings...")
        embeddings = embedding_model.encode(chunks)

        #3. Convert to numpy array (FAISS requires float32 numpy arrays)
        embeddings_np = np.array(embeddings).astype('float32')

        #4. Store in FAISS and document_store
        faiss_index.add(embeddings_np)
        document_store.extend(chunks)

        print(f"Successfully added {len(chunks)} vectors to FAISS. Total in index: {faiss_index.ntotal}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

    print(f"Finished processing: {filename}")

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(request: Request,background_tasks:BackgroundTasks,file:UploadFile =File(...)):
    #1. Save uploaded file locally 
    file_path = f"uploaded_docs/{file.filename}"
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    #2. Add the processing function ot the background tasks 
    background_tasks.add_task(process_document,file_path,file.filename)

    #3. Immediately return a response to the user
    return {
        "status" : "success",
        "message": f"Document '{file.filename}' received and is processing in the background."

    }
@app.get("/")
async def root():
    return {"message": "API is running!"}

#query endpoint
@app.post("/query")
@limiter.limit("10/minute")
async def query_document(request: Request,query_request: QueryRequest):
    #1. Check if the database is empty
    if faiss_index.ntotal == 0:
        return {"error":"No documents have been uploaded yet. Please upload a document first."}
    
    try:
        start_time = time.time()
        #2. Embed the question and search FAISS
        print(f"Searching for: '{query_request.question}'")
        question_embedding = embedding_model.encode([query_request.question])
        question_embedding_np = np.array(question_embedding).astype('float32')

        k = 3
        distances, indices = faiss_index.search(question_embedding_np,k)

        #3.Retrieve chunks
        retrieved_chunks = [document_store[i] for i in indices[0] if i != -1]
        if not retrieved_chunks:
            latency = time.time() - start_time
            return {
                "question": query_request.question,
                "answer": "I cannot answer this based on the provided documents.",
                "sources_used": 0,
                "latency":latency
                }

        # 4.Combine  retrieved text for the LLM
        context_text = " ".join(retrieved_chunks)
        
        # 5. Build strict RAG Prompt
        prompt = f"""
        You are a helpful assistant. Answer the user's question based ONLY on the following context. 
        If the answer cannot be found in the context, simply say "I cannot answer this based on the provided documents." Do not guess.

        Context:
        {context_text}

        Question: {query_request.question}
        """
        
        # 6. Generate the answer using Gemini!
        print("Sending context to Gemini...")
        response = llm.generate_content(prompt)
        
        latency = time.time() - start_time
        return {
            "question": query_request.question,
            "answer": response.text.strip(),
            "sources_used": len(retrieved_chunks),
            "latency" : latency
        }
        
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}