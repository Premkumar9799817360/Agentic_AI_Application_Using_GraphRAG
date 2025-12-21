import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

DATA_DIR = "Data"
GRAPH_FILE = "knowledge_graph.pkl"
MEMORY_FILE = "agent_memory.json"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")


client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer(EMBED_MODEL)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 20
SIMILARITY_THRESHOLD = 0.95
