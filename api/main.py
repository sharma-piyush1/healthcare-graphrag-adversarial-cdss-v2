import logging
import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List

# 1. LOAD DOTENV FIRST
from dotenv import load_dotenv
load_dotenv()

# 2. NOW IMPORT CORE MODULES
from core.graph_engine import v2_app
from core.vision_parser import VisionParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Graph-RAG API",
    description="Adversarial Clinical Decision Support System",
    version="2.0.0"
)

# Initialize the Vision Parser
vision_parser = VisionParser()

# --- API Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The clinical question from the user")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Recent conversation context")

class QueryResponse(BaseModel):
    generation: str
    source_type: str
    is_approved: bool
    audit_report: str
    retry_count: int

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "operational", "engine": "LangGraph v2"}

@app.post("/api/v1/clinical-query", response_model=QueryResponse)
def process_clinical_query(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    
    try:
        formatted_history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        initial_state = {
            "query": request.query,
            "chat_history": formatted_history,
            "retry_count": 0
        }
        
        result = v2_app.invoke(initial_state)
        
        return QueryResponse(
            generation=result.get("generation", "Error generating response."),
            source_type=result.get("source_type", "Unknown"),
            is_approved=result.get("is_approved", False),
            audit_report=result.get("audit_report", "N/A"),
            retry_count=result.get("retry_count", 0)
        )
        
    except Exception as e:
        logger.error(f"Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal CDSS Engine Error")

@app.post("/api/v1/analyze-report", response_model=QueryResponse)
async def process_clinical_report(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        vision_result = vision_parser.analyze_image(temp_file_path)
        
        if not vision_result.get("is_clinical", False):
            return QueryResponse(
                generation="The uploaded image does not appear to be a valid clinical document.",
                source_type="Vision Guardrail",
                is_approved=True,
                audit_report="Rejected: Non-clinical image.",
                retry_count=0
            )
            
        synthetic_query = f"Based on the uploaded report showing {vision_result['summary']}, what are the guidelines and treatments for: {', '.join(vision_result['concepts'])}?"
        
        initial_state = {
            "query": synthetic_query,
            "chat_history": [],
            "is_clinical": True,
            "concepts": vision_result['concepts'],
            "retry_count": 0
        }
        
        result = v2_app.invoke(initial_state)
        
        return QueryResponse(
            generation=result.get("generation", "Error generating response."),
            source_type=f"Multimodal + {result.get('source_type', 'Unknown')}",
            is_approved=result.get("is_approved", False),
            audit_report=result.get("audit_report", "N/A"),
            retry_count=result.get("retry_count", 0)
        )
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) 