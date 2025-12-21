import os
import json
import re
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from Config import DATA_DIR, embed_model, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, SIMILARITY_THRESHOLD


def load_documents(data_dir):
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                metadata = {
                    "filename": f,
                    "path": path,
                    "type": f.split('.')[-1],
                    "size": os.path.getsize(path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                }
                
                if f.endswith(".csv"):
                    df = pd.read_csv(path)
                    text = df.to_string()
                    metadata["rows"] = len(df)
                    metadata["columns"] = list(df.columns)
                elif f.endswith(".json"):
                    with open(path) as jf:
                        jdata = json.load(jf)
                    text = json.dumps(jdata)
                    metadata["keys"] = list(jdata.keys()) if isinstance(jdata, dict) else []
                elif f.endswith(".pdf"):
                    reader = PdfReader(path)
                    text = " ".join(p.extract_text() or "" for p in reader.pages)
                    metadata["pages"] = len(reader.pages)
                elif f.endswith(".txt"):
                    with open(path) as tf:
                        text = tf.read()
                else:
                    continue
                    
                docs.append({"text": text, "metadata": metadata})
            except Exception as e:
                st.warning(f"Failed to load {f}: {str(e)}")
    return docs


def clean_text(text):
    
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-Z0-9.,%$()\-/: ]", "", text)
    return text.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
   
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) > MIN_CHUNK_SIZE:  
            chunks.append(chunk)
    return chunks


def deduplicate_chunks(chunks):
    """Remove duplicate chunks based on similarity"""
    if not chunks:
        return chunks
    
    embeddings = embed_model.encode([c["text"] for c in chunks])
    unique_chunks = []
    used_indices = set()
    
    for i, chunk in enumerate(chunks):
        if i in used_indices:
            continue
        unique_chunks.append(chunk)
        used_indices.add(i)
        
       
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        for j, sim in enumerate(sims):
            if j > i and sim > SIMILARITY_THRESHOLD:  
                used_indices.add(j)
    
    return unique_chunks


@st.cache_resource
def preprocess():
    """Main preprocessing pipeline"""
    docs = load_documents(DATA_DIR)
    st.info(f"Loaded {len(docs)} documents")
    
    chunks = []
    for d in docs:
        clean = clean_text(d["text"])
        for c in chunk_text(clean):
            chunks.append({
                "text": c,
                "metadata": d["metadata"]
            })
    
    st.info(f"Created {len(chunks)} chunks")
    
    
    chunks = deduplicate_chunks(chunks)
    st.info(f"After deduplication: {len(chunks)} chunks")
    
    
    embeddings = embed_model.encode([c["text"] for c in chunks], show_progress_bar=True)
    
    return chunks, embeddings
