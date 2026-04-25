import os
import uuid
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
load_dotenv()

# --- 1. MOCK CLINICAL DATASET ---
# In production, replace this with pandas reading a CSV or JSON from S3/local.
clinical_knowledge = [
    {
        "disease": "Hypertension",
        "guideline": "Target BP < 130/80 mmHg",
        "treatments": ["Lisinopril", "Losartan", "Amlodipine"],
        "contraindications": []
    },
    {
        "disease": "Type 2 Diabetes",
        "guideline": "Target HbA1c < 7.0%",
        "treatments": ["Metformin", "Glipizide", "Tirzepatide"],
        "contraindications": ["Severe Renal Impairment (for Metformin)"]
    },
    {
        "disease": "Chronic Kidney Disease",
        "guideline": "Monitor eGFR and adjust renally cleared medications",
        "treatments": ["Dapagliflozin"],
        "contraindications": ["Ibuprofen", "Naproxen"]
    }
]

class ClinicalDataIngestor:
    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "securepassword123"))
        )
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.collection_name = "medical_entities"
        
        self._init_vector_db()

    def _init_vector_db(self):
        """Recreates the Qdrant collection for a clean slate."""
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            logging.info(f"Cleared existing Qdrant collection: {self.collection_name}")
            
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE) # size for all-MiniLM-L6-v2
        )

    def process_data(self):
        """Synchronizes data into both Neo4j and Qdrant."""
        points = []
        
        with self.neo4j_driver.session() as session:
            # Clear existing graph data for a clean test environment
            session.run("MATCH (n) DETACH DELETE n")
            logging.info("Cleared existing Neo4j graph.")

            for entry in clinical_knowledge:
                disease = entry["disease"]
                guideline = entry["guideline"]
                
                # Create Disease Node
                session.run(
                    "MERGE (d:Disease {name: $name}) SET d.clinical_guideline = $guideline",
                    name=disease, guideline=guideline
                )
                points.append(self._create_point(disease, "Disease"))

                # Process Treatments
                for drug in entry["treatments"]:
                    session.run(
                        """
                        MERGE (d:Disease {name: $disease_name})
                        MERGE (dr:Drug {name: $drug_name})
                        MERGE (dr)-[:TREATS]->(d)
                        """,
                        disease_name=disease, drug_name=drug
                    )
                    points.append(self._create_point(drug, "Drug"))
                
                # Process Contraindications
                for bad_drug in entry["contraindications"]:
                    session.run(
                        """
                        MERGE (d:Disease {name: $disease_name})
                        MERGE (dr:Drug {name: $drug_name})
                        MERGE (dr)-[:CONTRAINDICATED_FOR]->(d)
                        """,
                        disease_name=disease, drug_name=bad_drug
                    )
                    points.append(self._create_point(bad_drug, "Drug"))

        # Batch upload to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logging.info(f"Ingested {len(points)} entities into Qdrant.")
            
        logging.info("Data ingestion complete. Systems synchronized.")

    def _create_point(self, name: str, entity_type: str) -> PointStruct:
        """Helper to create a Qdrant point with embedded vector."""
        vector = self.embeddings.embed_query(name)
        return PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, name)), # Deterministic ID based on name
            vector=vector,
            payload={"name": name, "type": entity_type}
        )

if __name__ == "__main__":
    ingestor = ClinicalDataIngestor()
    ingestor.process_data()