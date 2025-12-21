import os
import json
import re
import pickle
import tempfile
import networkx as nx
import streamlit as st
from pyvis.network import Network
from Config import client, GROQ_MODEL, GRAPH_FILE

def extract_entities(text):

    prompt = f"""Extract financial entities and their relationships from the text.

Focus on: companies, markets, economic indicators, financial instruments, people, regulations.

Return ONLY valid JSON in this exact format:
{{
  "entities": ["Entity1", "Entity2"],
  "relations": [
    {{"source": "Entity1", "target": "Entity2", "relation": "affects", "confidence": 0.8}}
  ]
}}

Text: {text[:400]}

JSON:"""

    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        content = r.choices[0].message.content.strip()
        
        content = re.sub(r'^```json\s*|\s*```$', '', content)
        data = json.loads(content)
        
       
        if data.get("entities") or data.get("relations"):
            print(f"✓ Extracted {len(data.get('entities', []))} entities and {len(data.get('relations', []))} relations")
        
        return data
    except Exception as e:
        print(f"✗ Entity extraction error: {str(e)}")
        return {"entities": [], "relations": []}


@st.cache_resource
def build_graph(chunks):
    """Build knowledge graph with community detection"""
    
    if os.path.exists(GRAPH_FILE):
        try:
            with open(GRAPH_FILE, 'rb') as f:
                G = pickle.load(f)
                print(f"✓ Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                return G
        except Exception as e:
            print(f"✗ Failed to load graph, rebuilding: {str(e)}")
    
    print("Building new knowledge graph...")
    G = nx.DiGraph()
    
   
    progress_bar = st.progress(0)
    total = min(len(chunks), 50)  
    
    entities_count = 0
    relations_count = 0
    
    for idx, c in enumerate(chunks[:total]):
        try:
            data = extract_entities(c["text"])
            
     
            for e in data.get("entities", []):
                if e and len(e) > 2:
                    if G.has_node(e):
                        G.nodes[e]["frequency"] = G.nodes[e].get("frequency", 1) + 1
                    else:
                        G.add_node(e, frequency=1, type="entity")
                        entities_count += 1
            
            
            for r in data.get("relations", []):
                src = r.get("source")
                tgt = r.get("target")
                rel = r.get("relation", "related_to")
                conf = r.get("confidence", 0.5)
                
                if src and tgt and src != tgt:
                    if G.has_edge(src, tgt):
                        
                        G[src][tgt]["weight"] += conf
                        G[src][tgt]["frequency"] = G[src][tgt].get("frequency", 1) + 1
                    else:
                        G.add_edge(src, tgt, relation=rel, weight=conf, frequency=1)
                        relations_count += 1
            
            progress_bar.progress((idx + 1) / total)
        except Exception as e:
            print(f"✗ Error processing chunk {idx}: {str(e)}")
            continue
    
    print(f"✓ Graph built: {entities_count} entities, {relations_count} relations")
    print(f"✓ Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    
    
    if G.number_of_nodes() > 0:
        try:
            pagerank = nx.pagerank(G)
            nx.set_node_attributes(G, pagerank, "pagerank")
            print("✓ PageRank calculated")
        except Exception as e:
            print(f"✗ PageRank failed: {str(e)}")
    
   
    if G.number_of_nodes() > 0:
        try:
        
            G_undirected = G.to_undirected()
            communities = {}
            for i, component in enumerate(nx.connected_components(G_undirected)):
                for node in component:
                    communities[node] = i
            nx.set_node_attributes(G, communities, "community")
            print(f"✓ Community detection: {len(set(communities.values()))} communities")
        except Exception as e:
            print(f"✗ Community detection failed: {str(e)}")
        
            communities = {node: 0 for node in G.nodes()}
            nx.set_node_attributes(G, communities, "community")
    
 
    try:
        with open(GRAPH_FILE, 'wb') as f:
            pickle.dump(G, f)
        print(f"✓ Graph saved to {GRAPH_FILE}")
    except Exception as e:
        print(f"✗ Failed to save graph: {str(e)}")
    
    return G


def visualize_graph(G, query=""):

    
    if G.number_of_nodes() == 0:
        print("✗ Cannot visualize: Graph has no nodes")
        return None
    
    print(f"Visualizing graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    try:
        net = Network(height="500px", width="100%", directed=True, notebook=False)
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)
       
        communities = nx.get_node_attributes(G, "community")
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]
     
        for node in G.nodes():
            community_id = communities.get(node, 0)
            color = colors[community_id % len(colors)]
            size = 10 + G.nodes[node].get("frequency", 1) * 5
            pagerank = G.nodes[node].get('pagerank', 0)
            frequency = G.nodes[node].get('frequency', 1)
            
            title = f"{node}\nFrequency: {frequency}\nPageRank: {pagerank:.3f}"
            net.add_node(node, label=node, color=color, size=size, title=title)
        
    
        for u, v, d in G.edges(data=True):
            weight = d.get("weight", 0.5)
            relation = d.get("relation", "related")
            net.add_edge(u, v, label=relation, width=weight*2, title=f"{relation} (weight: {weight:.2f})")
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
        net.save_graph(tmp.name)
        tmp.close()
        
        print(f"✓ Graph visualization saved to {tmp.name}")
        return tmp.name
        
    except Exception as e:
        print(f"✗ Visualization error: {str(e)}")
        return None
