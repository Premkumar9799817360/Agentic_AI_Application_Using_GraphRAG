import os
import json
from datetime import datetime
from Config import MEMORY_FILE

class AgentMemory:
    def __init__(self):
        self.conversations = []
        self.user_preferences = {}
        self.query_history = []
        
    def add_interaction(self, query, answer, context):
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "context": context
        })
        self.query_history.append(query)
        
    def get_recent_context(self, n=3):
        return self.conversations[-n:] if len(self.conversations) >= n else self.conversations
        
    def save(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump({
                "conversations": self.conversations,
                "preferences": self.user_preferences,
                "history": self.query_history
            }, f, indent=2)
            
    def load(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                self.conversations = data.get("conversations", [])
                self.user_preferences = data.get("preferences", {})
                self.query_history = data.get("history", [])


            
