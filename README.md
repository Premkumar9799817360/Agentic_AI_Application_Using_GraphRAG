# Financial Analysis System with GraphRAG

An intelligent Financial Analysis system that combines Knowledge Graphs with Large Language Models using GraphRAG (Graph Retrieval-Augmented Generation) to answer complex financial queries with explainable reasoning.


## 🎯 Overview

I built an intelligent Financial Analysis system using GraphRAG (Graph Retrieval-Augmented Generation) that combines knowledge graphs with Large Language Models (LLMs) to answer complex financial queries with high accuracy, explainability, and multi-hop reasoning.

The system extracts entities and relationships from heterogeneous financial documents, constructs a knowledge graph, and performs hybrid retrieval (vector similarity + graph traversal) to generate grounded, explainable responses.

## ✨ This project demonstrates expertise in:

- **Knowledge Graph Construction**: Automatic entity and relationship extraction
- **Hybrid Retrieval**: Combines vector similarity and graph-based search
- **Agentic Workflow**: Multi-step autonomous reasoning
- **Chain-of-Thought**: Explainable step-by-step reasoning
- **Memory Management**: Conversation history and context preservation
- **Interactive UI**: Streamlit-based web interface with graph visualization

## 🛠️ Technologies & Skills

- Knowledge graph creation
- RAG pipeline design
- Agent workflow planning
- Prompt engineering
- Model integration (OpenAI(Groq)/Anthropic/Gemini)
- Document processing & embeddings
- API design
- End-to-end application architecture

## 📊 Project Architecture

![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/Architecture.png)


## 📁 Project Structure

```
financial-graphrag/
├── Data/
│   ├── csv_Data/          # 3 CSV files with stock prices and company data
│   ├── json_Data/         # 5 JSON files with financial data
│   ├── pdf_Data/          # 8 PDF files (research papers, reports)
│   └── textfile_data/     # 6 text files (articles, documentation)
├── Preprocessing.py       # Document loading, cleaning, chunking, embedding
├── GraphBuilding.py       # Knowledge graph construction and visualization
├── AgentWorkflow.py       # Agentic AI workflow and reasoning engine
├── AgentMemory.py         # Memory management and conversation history
├── config.py              # Configuration (API keys, model settings)
├── app.py                 # Streamlit UI application
├── knowledge_graph.pkl    # Serialized knowledge graph
└── agent_memory.json      # Persistent memory storage
```

---
## 🌐 Live Demo

You can test the project live using the link below:

👉 **[Live Testing URL](https://agenticaiapplicationusinggraphrag.streamlit.app/)**  

Or click the button below 👇  

<p align="center">
  <a href="[https://agripredict-9lbe.onrender.com/](https://agenticaiapplicationusinggraphrag.streamlit.app/)" target="_blank">
    <img src="https://img.shields.io/badge/🔗%20Open%20Live%20App-00C853?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Live App Button"/>
  </a>
</p>
---


# 1. Data Collection & Knowledge Base Construction

## Domain Selection

I selected the **Finance domain**  because it is highly impacted by current AI advancements and requires reasoning over structured + unstructured data, making it ideal for GraphRAG.

This domain also benefits significantly from:
- Multi-hop reasoning  
- Explainable answers  
- Relationship-based insights (e.g., company → market → indicators)

---

## Data Sources (20–30 Documents)
I collected and curated financial data from multiple sources:

- **Research papers** related to finance and economics  
- **Financial reports** in PDF format  
- **Finance blogs** and online articles  
- **Wikipedia articles**, converted into plain text files  
- **Kaggle datasets** containing stock prices and company-related financial data (CSV format)  
- **JSON-based financial datasets**  

To ensure efficient experimentation and reduce processing time during early development, CSV datasets were intentionally limited to **100–200 rows**.
![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/STORAGE%20LAYER.png)
```
## Dataset Structure

Data/
├── csv_Data/        (3 CSV files)
├── json_Data/       (5 JSON files)
├── pdf_Data/        (8 PDF files)
└── textfile_data/   (6 TXT files)
```


# 2. Document Preprocessing Pipeline
Implemented a robust preprocessing pipeline to standardize heterogeneous data formats.

![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/Preprocessing%20Layer.png)

## Key Steps
- **Text cleaning** (regex-based normalization)
- **Metadata extraction** (source, file type, timestamps)
- **Chunking with overlap**
- **Deduplication of similar chunks**
- **Embedding generation**

## Tools & Libraries

```
pandas (CSV & JSON handling)
PyPDF2 (PDF parsing)
re (text cleaning)
sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
```
## Output
- Cleaned, deduplicated, embedded document chunks
- Stored for both vector retrieval and graph construction
```
File: Preprocessing.py
```

# 3. Knowledge Graph Construction (GraphRAG)

![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/Knowledge%20Layer.png)

## Graph Design
I constructed a semantic knowledge graph using NetworkX, enriched with entity relationships extracted via LLM prompts.

## Graph Components
- Nodes: Financial entities (companies, metrics, concepts, events)
- Edges: Relationships between entities
- Weights: Confidence scores from extraction


## Process
- Entity extraction from text chunks using Groq LLM
- Relationship identification via prompt-based extraction
- Graph creation using NetworkX
- Visualization using PyVis, dynamically filtered by user query

## Visualization
The graph visualization displays:
- Relevant nodes
- Relationship paths
- Multi-hop connections tied to the query
  
```
File: GraphBuilding.py
```

# 4. Agentic Workflow Design
I designed a multi-step autonomous agent workflow to enable intelligent reasoning.

This is the main file responsible for the agentic AI workflow, which defines how the entire project operates. It includes extensive prompt engineering and manual intent classification.

I created predefined intent categories such as factual_query and multi_hop_reasoning to guide the agent’s behavior. Using classes and methods, the system identifies the user’s intent and routes the query through appropriate processing steps.

The workflow implements multiple functions, including:
- Graph-based retrieval
- Multi-hop retrieval using the knowledge graph
- Chain-of-thought generation for reasoning
- Answer generation

Additionally, an answer evaluation module is implemented to assess the quality of the response. This evaluation checks the answer length, verifies whether the answer is supported by source documents, and calculates a confidence score. If the confidence score is 50% or above, the result is marked as High confidence; otherwise, it is marked as Medium confidence.

![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/Application%20Layer.png)

## Agent Capabilities
- User intent understanding
- Hybrid retrieval (vector + graph)
- Multi-hop reasoning over graph paths
- Context selection
- Answer grounding & synthesis
- Confidence evaluation
  
## Intent Classification
Manually defined intent categories:

```
1. factual_query
2. multi_hop_reasoning
3. explanatory_query
```

```
File: AgentWorkflow.py
```
# 5. Agent Memory System

I implemented a manual agent memory mechanism to store previous interactions in a JSON file, which is then loaded into the Streamlit UI to maintain conversational context.

The AgentMemory.py file manages the memory system and defines three separate lists to store different types of information:
- Conversation history – stores past user–agent interactions
- User preferences – stores user-specific settings or behavior patterns
- Query history – tracks previously asked queries

Each conversation record is stored with a timestamp, along with the user query, the generated answer, and the retrieved context. This structured memory design enables context-aware responses, improves follow-up question handling, and provides a more personalized user experience.

```
File: AgentMemory.py
```

# 6. Configuration & Model Integration
A main configuration file is used to centrally manage all LLM models, API keys, and system parameters required by the application.

## Model Configuration
- LLM (via Groq): openai/gpt-oss-120b
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2

## Chunking & Retrieval Settings
The configuration file also defines key preprocessing and retrieval parameters, including:
- Chunk size
- Chunk overlap
- Minimum chunk size
- Similarity threshold

These settings control document chunking, embedding quality, and retrieval accuracy.

## Persistent Storage Files
The system uses the following persistent files:
```
1. knowledge_graph.pkl – Stores the serialized knowledge graph
2. agent_memory.json – Stores agent memory data in list format
```
Centralizing these configurations ensures easy maintenance, model flexibility, and consistent behavior across the entire GraphRAG pipeline.

```
File: Config.py
```

# 7. Streamlit User Interface

The Streamlit UI is the final layer of the system and displays all important details of the GraphRAG application. It provides a simple chat interface that allows users to ask questions and interact with the agent.

![GraphRAG Architecture](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/Images/Presentation%20Layer.png)



The UI includes the following features:
- A **chat interface** for querying the agent
- **Graph visualization**, showing related nodes, relationships, and paths based on the user’s query
- Display of **retrieved source documents or text chunks**, helping users understand where the answer comes from

Although the UI is minimal, it clearly demonstrates the end-to-end functionality of the GraphRAG system.

## UI Options
 ### Show Chain of Thought
This option explains how the LLM reasons step by step to find the best answer.

 ### Show Evaluation Metrics
 - Displays:
   - Whether the answer is supported by source documents
   - Answer length
   - Confidence score
     - High confidence: 50% or above
     - Medium confidence: below 50%
###  Graph Statistics
Shows system statistics such as:
- Number of documents
- Number of chunks
- Number of graph nodes
- Number of graph edges
- Average graph metrics

### Conversation History
Displays the last three user conversations for quick reference.

### System Controls
Reload system when settings are changed
Clear cache option to reset the session

This Streamlit interface makes the system easy to use, interactive, and transparent, helping users understand both the answers and the reasoning process behind them.

```
File: Streamlit.py
```

# 📸 Project Screenshots

## 📌 1. Home Page

### Project User Interface Overview

![HomePage](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Home_Page.png)


## 📌 2. Answer of Question (GraphRAG + LLM Response)

### Intelligent Question Answering using GraphRAG & LLM

![HomePage](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Home_Page.png)


## 📌 3. Knowledge Graph & Retrieval Source

### Knowledge Graph Visualization & Context Retrieval Sources

![Knowledge_graph](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Knowledge_Graph_And_Retrieval_Source.png)


## 📌 4. Chain-of-Thought (CoT)

### Multi-Step Reasoning & Chain-of-Thought Explanation

![COT](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Chain-of-Thought.png)

## 📌 5. Evaluation Metrics

### Model Performance & Evaluation Metrics Analysis

![Evaluation Metrics](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Evaluation_Metrics.png)


### 📌 6. Previous History

### User Query History & Past Interactions

![History](https://github.com/Premkumar9799817360/Agentic_AI_Application_Using_GraphRAG_MindGraph/blob/main/ProjectImage/Past_Interactions_History.png)


