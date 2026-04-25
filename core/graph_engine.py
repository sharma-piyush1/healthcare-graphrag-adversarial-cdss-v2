import os
import logging
from typing import TypedDict, List
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# --- 1. STATE SCHEMA ---
class GraphRAGState(TypedDict):
    query: str
    chat_history: List[dict]
    is_clinical: bool            
    concepts: List[str]
    retrieved_context: str
    source_type: str
    generation: str
    audit_report: str
    is_approved: bool
    retry_count: int

# --- 2. INFRASTRUCTURE INIT ---
# Primary Generator
gen_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
# Auditor / Red Team
audit_llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "securepassword123"))
)
qdrant_client = QdrantClient("localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
COLLECTION_NAME = "medical_entities"

# --- 3. STRICT SCHEMA ENFORCEMENT ---
class IntentClassification(BaseModel):
    is_clinical: bool = Field(description="True ONLY if the query is explicitly about human medicine, pharmacology, or clinical operations.")
    confidence_score: float = Field(description="0.0 to 1.0 score of how clinical the query is.")
    concepts: List[str] = Field(description="Extracted medical entities. Empty if is_clinical is false.")
    reasoning: str = Field(description="One sentence explaining why it is or isn't clinical.")

# --- 4. GRAPH NODES ---

def extract_node(state: GraphRAGState):
    logging.info("NODE: Strict Semantic Gatekeeper")
    
    prompt = """
    You are a strict Medical Triage AI. Classify the user query.
    
    RULES:
    - General knowledge, chit-chat, coding, or fictional queries = FALSE.
    - If a query mixes a non-medical topic with a medical word = FALSE.
    - Only authorize legitimate clinical inquiries.
    
    EXAMPLES:
    Q: "What is the capital of France?" -> is_clinical: false
    Q: "How does Losartan work?" -> is_clinical: true
    Q: "what is the medicine of Hypertension" -> is_clinical: true
    
    Chat History: {history}
    Query: {query}
    """
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.get("chat_history", [])[-4:]])
    structured_llm = gen_llm.with_structured_output(IntentClassification)
    
    try:
        chain = ChatPromptTemplate.from_template(prompt) | structured_llm
        result = chain.invoke({"query": state["query"], "history": history_str})
        
        # Safely handle both Pydantic objects and Dictionaries
        if isinstance(result, dict):
            is_clin = result.get("is_clinical", False)
            conf = result.get("confidence_score", 0.0)
            extracted_concepts = result.get("concepts", [])
            reasoning = result.get("reasoning", "")
        else:
            is_clin = getattr(result, "is_clinical", False)
            conf = getattr(result, "confidence_score", 0.0)
            extracted_concepts = getattr(result, "concepts", [])
            reasoning = getattr(result, "reasoning", "")
        
        # Must pass boolean flag AND confidence check (lowered to 0.7 for brief queries)
        is_clinical = is_clin and conf >= 0.7
        concepts = extracted_concepts if is_clinical else []
        
        logging.info(f"Gatekeeper Logic: {reasoning} | Confidence: {conf}")
        
    except Exception as e:
        logging.error(f"Classification failed, defaulting to reject. Error: {e}")
        is_clinical = False
        concepts = []
        
    return {"is_clinical": is_clinical, "concepts": concepts}

def reject_node(state: GraphRAGState):
    """Triggered when the query is off-topic to save tokens."""
    logging.warning("NODE: Query Rejected. Off-Topic.")
    return {
        "generation": "I am a Clinical Decision Support System. I am restricted to answering medical, pharmacological, and healthcare-related queries. I cannot assist with general topics or casual conversation.",
        "source_type": "System Guardrail",
        "is_approved": True, 
        "audit_report": "Bypassed (Out of Domain)",
        "retrieved_context": ""
    }

def retrieve_node(state: GraphRAGState):
    logging.info("NODE: Hybrid Graph Retrieval (With Alternative Logic)")
    context = []
    
    with neo4j_driver.session() as session:
        for concept in state["concepts"]:
            vector = embeddings.embed_query(concept)
            search_result = qdrant_client.search(
                collection_name=COLLECTION_NAME, 
                query_vector=vector, 
                limit=1
            )
            
            if not search_result:
                continue
                
            matched_node = search_result[0].payload
            official_name = matched_node['name']
            
            # Standard Query
            std_query = """
            MATCH (n {name: $name})-[r]-(m) 
            RETURN n.name AS source, type(r) AS rel, m.name AS target, n.clinical_guideline AS guideline
            """
            std_result = session.run(std_query, name=official_name)
            
            for record in std_result:
                guideline = f" (Guideline: {record['guideline']})" if record['guideline'] else ""
                context.append(f"{record['source']} {record['rel']} {record['target']}{guideline}")
            
            # Alternative Discovery
            alt_query = """
            MATCH (d:Drug {name: $name})-[:TREATS]->(condition:Disease)<-[:TREATS]-(alt:Drug)
            RETURN alt.name AS alternative, condition.name AS condition
            """
            alt_result = session.run(alt_query, name=official_name)
            
            for record in alt_result:
                context.append(f"ALTERNATIVE DRUG: '{record['alternative']}' also treats '{record['condition']}'.")
    
    context_str = "\n".join(set(context))
    if not context_str.strip():
        return {"retrieved_context": "", "source_type": "None"}
        
    return {"retrieved_context": context_str, "source_type": "Local Graph"}

def research_node(state: GraphRAGState):
    logging.info("NODE: Graph Empty. Triggering Web Fallback.")
    try:
        search = tavily_client.search(query=state["query"], search_depth="advanced", max_results=3)
        web_context = "\n".join([f"Source ({r['url']}): {r['content']}" for r in search['results']])
        return {"retrieved_context": f"WEB RESEARCH:\n{web_context}", "source_type": "Live Web Search"}
    except Exception as e:
        return {"retrieved_context": f"Error during web search: {str(e)}", "source_type": "Error"}

def generate_node(state: GraphRAGState):
    logging.info("NODE: Generating Clinical Response")
    sys_msg = "You are a clinical AI. Answer using ONLY the provided context."
    if state.get("audit_report"):
        sys_msg += f"\n\nPRIOR AUDIT FAILURE - FIX THIS IMMEDIATELY: {state['audit_report']}"
    
    prompt = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "Query: {q}\n\nContext:\n{c}")])
    res = (prompt | gen_llm).invoke({"q": state["query"], "c": state["retrieved_context"]})
    return {"generation": res.content, "retry_count": state.get("retry_count", 0) + 1}

def audit_node(state: GraphRAGState):
    logging.info("NODE: Red Team Audit (120B)")
    prompt = """AUDIT TASK: Compare the AI Response to the Context.
    1. Are there claims in the response NOT explicitly in the context?
    2. Did it miss a critical contraindication?
    Return strict JSON: {{"is_approved": true/false, "critique": "string feedback or 'Approved'"}}
    
    Context: {c}
    Response: {r}"""
    
    chain = ChatPromptTemplate.from_template(prompt) | audit_llm | JsonOutputParser()
    try:
        report = chain.invoke({"c": state["retrieved_context"], "r": state["generation"]})
    except Exception as e:
        report = {"is_approved": False, "critique": f"Audit parsing failed: {str(e)}"}
        
    logging.info(f"AUDIT RESULT: Approved={report['is_approved']}")
    return {"is_approved": report["is_approved"], "audit_report": report["critique"]}

# --- 5. ROUTING LOGIC ---

def route_gatekeeper(state: GraphRAGState):
    if not state.get("is_clinical", True):
        return "reject"
    return "retrieve"

def route_retrieval(state: GraphRAGState):
    if not state["retrieved_context"].strip():
        return "research"
    return "generate"

def route_audit(state: GraphRAGState):
    if state["is_approved"] or state["retry_count"] >= 3:
        return "end"
    return "rewrite"

# --- 6. BUILD GRAPH ---
builder = StateGraph(GraphRAGState)

builder.add_node("extract", extract_node)
builder.add_node("reject", reject_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("research", research_node)
builder.add_node("generate", generate_node)
builder.add_node("audit", audit_node)

builder.set_entry_point("extract")

builder.add_conditional_edges("extract", route_gatekeeper, {"retrieve": "retrieve", "reject": "reject"})
builder.add_edge("reject", END)

builder.add_conditional_edges("retrieve", route_retrieval, {"research": "research", "generate": "generate"})
builder.add_edge("research", "generate")

builder.add_edge("generate", "audit")
builder.add_conditional_edges("audit", route_audit, {"end": END, "rewrite": "generate"})

v2_app = builder.compile()