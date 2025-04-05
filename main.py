from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import cohere
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function for joining herb-phytochemical data
def get_related_compounds(herb_ids):
    hp_response = supabase.table("herb_phytochemical").select("*").in_("herb_id", herb_ids).execute()
    if not hp_response.data:
        return []
    
    compound_ids = [hp["compound_id"] for hp in hp_response.data]
    compounds_response = supabase.table("phytochemicals").select("*").in_("compound_id", compound_ids).execute()
    return compounds_response.data

# Helper function for joining phytochemical-herb data
def get_related_herbs(compound_ids):
    hp_response = supabase.table("herb_phytochemical").select("*").in_("compound_id", compound_ids).execute()
    if not hp_response.data:
        return []
    
    herb_ids = [hp["herb_id"] for hp in hp_response.data]
    herbs_response = supabase.table("herbs").select("*").in_("herb_id", herb_ids).execute()
    return herbs_response.data

@app.get("/herbs/{herb_name}")
async def get_herb(herb_name: str):
    # Get herbs with name match
    herb_response = supabase.table("herbs").select("*").ilike("herb_name", f"%{herb_name}%").execute()
    if not herb_response.data:
        raise HTTPException(status_code=404, detail="Herb not found")
    
    herbs = herb_response.data
    herb_ids = [h["herb_id"] for h in herbs]
    compounds = get_related_compounds(herb_ids)
    
    # Map compounds to herbs
    compound_map = {}
    for hp in supabase.table("herb_phytochemical").select("*").in_("herb_id", herb_ids).execute().data:
        compound = next((c for c in compounds if c["compound_id"] == hp["compound_id"]), None)
        if compound:
            compound_map.setdefault(hp["herb_id"], []).append({
                "compound_name": compound["compound_name"],
                "function": compound["function"],
                "compound_type": compound["compound_type"],
                "source_url": compound["source_url"]
            })
    
    return [{
        "herb_name": h["herb_name"],
        "scientific_name": h["scientific_name"],
        "uses": h["uses"],
        "origin": h["origin"],
        "source_url": h["source_url"],
        "phytochemicals": compound_map.get(h["herb_id"], [])
    } for h in herbs]

@app.get("/phytochemicals/{compound_name}")
async def get_phytochemical(compound_name: str):
    # Get compounds with name match
    compound_response = supabase.table("phytochemicals").select("*").ilike("compound_name", f"%{compound_name}%").execute()
    if not compound_response.data:
        raise HTTPException(status_code=404, detail="Compound not found")
    
    compounds = compound_response.data
    compound_ids = [c["compound_id"] for c in compounds]
    herbs = get_related_herbs(compound_ids)
    
    # Map herbs to compounds
    herb_map = {}
    for hp in supabase.table("herb_phytochemical").select("*").in_("compound_id", compound_ids).execute().data:
        herb = next((h for h in herbs if h["herb_id"] == hp["herb_id"]), None)
        if herb:
            herb_map.setdefault(hp["compound_id"], []).append({
                "herb_name": herb["herb_name"],
                "scientific_name": herb["scientific_name"],
                "uses": herb["uses"],
                "source_url": herb["source_url"]
            })
    
    return [{
        "compound_name": c["compound_name"],
        "function": c["function"],
        "chemical_structure": c["chemical_structure"],
        "compound_type": c["compound_type"],
        "source_url": c["source_url"],
        "related_herbs": herb_map.get(c["compound_id"], [])
    } for c in compounds]

@app.get("/search")
async def search(q: str):
    # Get all available data
    herbs = supabase.table("herbs").select("*").execute().data
    compounds = supabase.table("phytochemicals").select("*").execute().data
    
    # Prepare documents for reranking
    documents = []
    meta = []
    
    for h in herbs:
        text = f"Herb: {h['herb_name']}. Uses: {h['uses']}. Origin: {h['origin']}"
        documents.append(text)
        meta.append({
            "type": "herb",
            "name": h["herb_name"],
            "info": h["uses"],
            "source_url": h["source_url"]
        })
    
    for c in compounds:
        text = f"Compound: {c['compound_name']}. Function: {c['function']}. Type: {c['compound_type']}"
        documents.append(text)
        meta.append({
            "type": "compound",
            "name": c["compound_name"],
            "info": c["function"],
            "source_url": c["source_url"]
        })
    
    # Perform reranking
    try:
        rerank = co.rerank(
            query=q,
            documents=documents,
            top_n=5,
            model="rerank-english-v2.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere error: {str(e)}")
    
    # Compile results
    results = []
    for result in rerank.results:
        if result.relevance_score > 0.2:  # Minimum relevance threshold
            data = meta[result.index]
            results.append({
                "type": data["type"],
                "name": data["name"],
                "info": data["info"],
                "source_url": data["source_url"]
            })
    
    return results[:5]  # Return top 5 results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)