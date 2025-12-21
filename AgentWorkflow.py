# ================================
# AGENT WORKFLOW COMPONENTS
# ================================

import re
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from Config import GROQ_MODEL, client, embed_model


class IntentClassifier:
    """LLM-based intent classifier"""
    
    INTENTS = [
        "factual_query",
        "comparison",
        "trend_analysis",
        "multi_hop_reasoning",
        "definition",
        "explanation"
    ]
    
    @staticmethod
    def classify(query):
        prompt = f"""Classify the user's intent into ONE of these categories:
- factual_query: Simple fact retrieval
- comparison: Comparing two or more things
- trend_analysis: Analyzing trends or patterns
- multi_hop_reasoning: Requires connecting multiple concepts
- definition: Asking for a definition
- explanation: Asking how something works

Query: {query}

Intent (one word):"""
        
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            intent = r.choices[0].message.content.strip().lower()
            return intent if intent in IntentClassifier.INTENTS else "factual_query"
        except:
            return "factual_query"


def graph_aware_retrieval(G, query, k=5):
    """Graph-aware retrieval with node importance"""
    keywords = query.lower().split()
    node_scores = {}
    
    for node in G.nodes():
        score = 0
        node_lower = node.lower()
        
        # Keyword matching
        for kw in keywords:
            if kw in node_lower:
                score += 2
        
        # PageRank bonus
        score += G.nodes[node].get("pagerank", 0) * 10
        
        # Frequency bonus
        score += G.nodes[node].get("frequency", 0) * 0.1
        
        if score > 0:
            node_scores[node] = score
    
    # Get top nodes
    top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Get subgraph with neighbors
    nodes_to_include = set()
    for node, _ in top_nodes:
        nodes_to_include.add(node)
        nodes_to_include.update(G.predecessors(node))
        nodes_to_include.update(G.successors(node))
    
    return G.subgraph(nodes_to_include).copy()


def multi_hop_reasoning(G, start_node, end_node, max_hops=3):
    """Find paths between entities for multi-hop reasoning"""
    try:
        paths = []
        for path in nx.all_simple_paths(G, start_node, end_node, cutoff=max_hops):
            paths.append(path)
            if len(paths) >= 3:  # Limit paths
                break
        return paths
    except:
        return []


def retrieve_with_graph(query, chunks, embeddings, G, k=5):
    """Hybrid retrieval: vector + graph"""
    # Vector retrieval
    q_emb = embed_model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_indices = np.argsort(sims)[-k:][::-1]
    vector_chunks = [chunks[i] for i in top_indices]
    
    # Graph retrieval
    subgraph = graph_aware_retrieval(G, query, k)
    
    return vector_chunks, subgraph


def generate_chain_of_thought(query, context, graph_info):
    """Hidden chain-of-thought reasoning"""
    cot_prompt = f"""Think step by step about this query:

Query: {query}

Context: {context[:500]}

Graph relationships: {graph_info[:300]}

Break down your reasoning:
1. What is being asked?
2. What information is relevant?
3. How do the pieces connect?
4. What's the final answer?

Reasoning:"""
    
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return r.choices[0].message.content
    except:
        return "Unable to generate reasoning chain."


def generate_answer(query, context, graph_info, reasoning, memory_context=""):
    """Final answer generation with all context"""
    prompt = f"""You are a financial AI assistant. Use the provided context and graph relationships to answer accurately.

Question: {query}

Context from documents:
{context}

Graph relationships:
{graph_info}

Internal reasoning:
{reasoning}

Previous conversation context:
{memory_context}

Provide a clear, accurate answer based on the information above. Cite sources when possible.

Answer:"""
    
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def evaluate_answer(query, answer, ground_truth=None):
    """Basic evaluation metrics"""
    metrics = {
        "answer_length": len(answer.split()),
        "has_numbers": bool(re.search(r'\d', answer)),
        "has_sources": bool(re.search(r'according to|source|based on', answer.lower())),
        "confidence": "high" if len(answer.split()) > 50 else "medium"
    }
    return metrics