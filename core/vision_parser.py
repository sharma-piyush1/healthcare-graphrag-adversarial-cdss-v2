import base64
import logging
import json
from typing import List, Union, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# --- RELAXED SCHEMA TO BYPASS GROQ VALIDATOR ---
class ClinicalImageExtraction(BaseModel):
    # We use 'Any' or 'str' because Groq/Llama often returns strings for booleans/arrays
    is_clinical_image: Any = Field(description="Set to 'true' if image contains medical text. String or Boolean accepted.")
    concepts: Any = Field(description="List of medical entities. Stringified list or Array accepted.")
    summary: str = Field(description="A brief summary of the text.")

class VisionParser:
    def __init__(self):
        self.vision_llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0
        )
        # Use 'json_mode' if tool_use continues to fail strict validation
        self.structured_llm = self.vision_llm.with_structured_output(ClinicalImageExtraction)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str) -> dict:
        logger.info(f"Analyzing multimodal input: {image_path}")
        base64_image = self.encode_image(image_path)
        
        prompt_text = "Extract medical entities from this image. Return 'is_clinical_image' as true if medical text is found."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        
        try:
            result = self.structured_llm.invoke([message])
            
            # --- ROBUST TYPE CONVERSION ---
            # 1. Parse is_clinical_image
            raw_is_clin = getattr(result, "is_clinical_image", False) if not isinstance(result, dict) else result.get("is_clinical_image", False)
            if isinstance(raw_is_clin, str):
                is_clin = raw_is_clin.lower() in ["true", "1", "yes"]
            else:
                is_clin = bool(raw_is_clin)

            # 2. Parse concepts (handles stringified lists like '["Drug"]')
            raw_concepts = getattr(result, "concepts", []) if not isinstance(result, dict) else result.get("concepts", [])
            if isinstance(raw_concepts, str):
                try:
                    # Try to parse if it's a JSON-style string
                    concepts = json.loads(raw_concepts.replace("'", '"'))
                except:
                    # Fallback for comma-separated messy strings
                    concepts = [c.strip() for c in raw_concepts.strip("[]").replace('"', '').split(",") if c.strip()]
            else:
                concepts = raw_concepts

            summary = getattr(result, "summary", "") if not isinstance(result, dict) else result.get("summary", "")

            # Override Logic
            force_clinical = is_clin or len(concepts) > 0
            logger.info(f"Vision Final -> Clinical: {force_clinical}, Concepts: {concepts}")

            return {"is_clinical": force_clinical, "concepts": concepts, "summary": summary}
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}")
            return {"is_clinical": False, "concepts": [], "summary": f"Error: {str(e)}"}