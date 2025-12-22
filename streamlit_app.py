
import networkx as nx
import streamlit as st
from AgentMemory import AgentMemory
from Preprocessing import preprocess
from GraphBuilding import build_graph, visualize_graph
from AgentWorkflow import (
    IntentClassifier,
    retrieve_with_graph,
    generate_chain_of_thought,
    generate_answer,
    evaluate_answer
)


def main():
    st.set_page_config("Finance GraphRAG", layout="wide", page_icon="📊")
    
    
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: #4ECDC4;
            color: white;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Finance GraphRAG Assistant")
    st.markdown("*Intelligent Financial Analysis with Knowledge Graphs*")
    

    if 'memory' not in st.session_state:
        st.session_state.memory = AgentMemory()
        st.session_state.memory.load()

    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_reasoning = st.checkbox("Show Chain-of-Thought", value=False)
        show_metrics = st.checkbox("Show Evaluation Metrics", value=False)
        show_graph_stats = st.checkbox("Show Graph Statistics", value=True)
        k_chunks = st.slider("Retrieved Chunks", 3, 10, 5)
        
        st.header("📈 System Stats")
        if st.button("Reload System"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("Clear Graph Cache"):
            import os
            from Config import GRAPH_FILE
            if os.path.exists(GRAPH_FILE):
                os.remove(GRAPH_FILE)
                st.success("Graph cache cleared!")
                st.cache_resource.clear()
                st.rerun()
    

    with st.spinner("🔄 Loading knowledge base..."):
        try:
            chunks, embeddings = preprocess()
            G = build_graph(chunks)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📄 Documents", len(set(c["metadata"]["filename"] for c in chunks)))
            col2.metric("🧩 Chunks", len(chunks))
            col3.metric("🕸️ Graph Nodes", G.number_of_nodes())
            
            
            if show_graph_stats:
                col4, col5, col6 = st.columns(3)
                col4.metric("🔗 Graph Edges", G.number_of_edges())
                col5.metric("📊 Avg Degree", f"{sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1):.2f}")
                col6.metric("🌐 Connected", "Yes" if nx.is_weakly_connected(G) else "No")
            
            
            if G.number_of_nodes() == 0:
                st.warning("⚠️ Knowledge graph is empty. The system may not provide graph-based insights.")
            elif G.number_of_edges() == 0:
                st.warning("⚠️ Knowledge graph has nodes but no relationships. Graph reasoning will be limited.")
                
        except Exception as e:
            st.error(f"❌ Error loading system: {str(e)}")
            return
    

    st.header("💬 Ask a Question")
    
   
    if st.session_state.memory.conversations:
        with st.expander("📜 Conversation History"):
            for conv in st.session_state.memory.conversations[-5:]:
                st.markdown(f"**Q:** {conv['query']}")
                st.markdown(f"**A:** {conv['answer'][:200]}...")
                st.markdown("---")
    
    query = st.chat_input("Enter your financial question...")
    
    if query:
        with st.spinner("🤔 Processing your question..."):
            try:
                
                intent = IntentClassifier.classify(query)
                st.info(f"🎯 Detected Intent: **{intent}**")
                
             
                vector_chunks, subgraph = retrieve_with_graph(query, chunks, embeddings, G, k_chunks)
                
                
                st.write(f"🔍 Retrieved subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
                
               
                context = "\n\n".join([f"[{c['metadata']['filename']}]: {c['text'][:300]}" 
                                       for c in vector_chunks])
                
             
                if subgraph.number_of_edges() > 0:
                    graph_info = "\n".join([
                        f"• {u} --[{d['relation']}]--> {v} (confidence: {d['weight']:.2f})"
                        for u, v, d in subgraph.edges(data=True)
                    ])
                else:
                    graph_info = "No direct graph relationships found for this query."
         
                reasoning = generate_chain_of_thought(query, context, graph_info)
                
             
                memory_ctx = ""
                if st.session_state.memory.conversations:
                    recent = st.session_state.memory.get_recent_context(2)
                    memory_ctx = "\n".join([f"Previous Q: {c['query']}\nA: {c['answer'][:100]}" 
                                           for c in recent])
                
                
                answer = generate_answer(query, context, graph_info, reasoning, memory_ctx)
                
              
                st.session_state.memory.add_interaction(query, answer, context)
                st.session_state.memory.save()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        
        st.subheader("🧠 Answer")
        st.markdown(f"<div class='metric-card'>{answer}</div>", unsafe_allow_html=True)
        
        
        if show_reasoning:
            with st.expander("🔍 Chain-of-Thought Reasoning"):
                st.write(reasoning)
        
      
        if show_metrics:
            metrics = evaluate_answer(query, answer)
            with st.expander("📊 Answer Metrics"):
                cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    cols[i].metric(k.replace("_", " ").title(), v)
        
      
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🕸️ Knowledge Graph")
            
           
            if subgraph.number_of_nodes() > 0:
                st.write(f"Nodes: {subgraph.number_of_nodes()}, Edges: {subgraph.number_of_edges()}")
                
               
                html_path = visualize_graph(subgraph, query)
                
                if html_path:
                    try:
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=520, scrolling=True)
                    except Exception as e:
                        st.error(f"Failed to display graph: {str(e)}")
                else:
                    st.warning("⚠️ Failed to generate graph visualization")
            else:
                st.info("ℹ️ No relevant graph connections found for this query.")
                
                if G.number_of_nodes() > 0 and G.number_of_nodes() <= 30:
                    with st.expander("Show Full Knowledge Graph"):
                        html_path = visualize_graph(G, query)
                        if html_path:
                            with open(html_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=520, scrolling=True)
        
        with col2:
            st.subheader("📄 Retrieved Sources")
            for i, chunk in enumerate(vector_chunks[:3]):
                with st.expander(f"Source {i+1}: {chunk['metadata']['filename']}"):
                    st.write(chunk['text'][:500] + "...")
                    st.json(chunk['metadata'])
            
            
            if subgraph.number_of_nodes() > 0:
                with st.expander("🏷️ Extracted Entities"):
                    entities = list(subgraph.nodes())
                    st.write(", ".join(entities[:20]))
                    if len(entities) > 20:
                        st.write(f"... and {len(entities) - 20} more")


if __name__ == "__main__":
    main()
