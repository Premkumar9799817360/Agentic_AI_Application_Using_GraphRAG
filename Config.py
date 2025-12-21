import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directories and Files
DATA_DIR = "Data"
GRAPH_FILE = "knowledge_graph.pkl"
MEMORY_FILE = "agent_memory.json"

# API Configuration (SAFE)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Validate key
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables")

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer(EMBED_MODEL)

# Chunking parameters
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 20
SIMILARITY_THRESHOLD = 0.95
