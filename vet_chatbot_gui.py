import os
import json
import faiss
import numpy as np
import uuid
import re
import threading
import requests
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk

# GUI Application Class
class VeterinaryChatApp:
    def __init__(self, root, chatbot):
        self.root = root
        self.chatbot = chatbot
        self.root.title("Veterinary Expert Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f8ff')  # Light blue background
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f8ff')
        self.style.configure('TLabel', background='#f0f8ff', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        
        # Create header
        header_frame = ttk.Frame(root)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Try to load logo
        try:
            logo_img = Image.open("vet_logo.png").resize((60, 60))
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header_frame, image=self.logo)
            logo_label.pack(side=tk.LEFT, padx=5)
        except:
            pass  # Continue without logo if file not found
        
        title_label = ttk.Label(header_frame, 
                               text="Veterinary Expert Assistant", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c6fbb')
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Create chat display area
        chat_frame = ttk.Frame(root)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            font=('Arial', 11),
            bg='white',
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Create input area
        input_frame = ttk.Frame(root)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.user_input = tk.Text(
            input_frame, 
            height=3, 
            font=('Arial', 11),
            bg='white'
        )
        self.user_input.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.user_input.bind("<Return>", self.on_enter_pressed)
        
        send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message,
            style='TButton'
        )
        send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add welcome message
        self.add_message("Assistant", "Hello! I'm your veterinary expert assistant. How can I help today?")
        
        # Set focus to input field
        self.user_input.focus_set()
    
    def on_enter_pressed(self, event):
        """Handle Enter key press (without Shift)"""
        if not event.state & 0x0001:  # Check if Shift is not pressed
            self.send_message()
            return "break"  # Prevent default Enter behavior
        return None
    
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Configure tags for styling
        self.chat_display.tag_config('assistant', foreground='#2c6fbb', font=('Arial', 11, 'bold'))
        self.chat_display.tag_config('user', foreground='#2e8b57', font=('Arial', 11, 'bold'))
        self.chat_display.tag_config('warning', foreground='#b22222')
        self.chat_display.tag_config('reference', foreground='#6a5acd')
        
        # Add sender label
        self.chat_display.insert(tk.END, f"{sender}: ", 'assistant' if sender == "Assistant" else 'user')
        
        # Add message content
        self.chat_display.insert(tk.END, message + "\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.yview(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self):
        """Send user message to chatbot"""
        user_input = self.user_input.get("1.0", tk.END).strip()
        if not user_input:
            return
            
        self.user_input.delete("1.0", tk.END)
        self.add_message("You", user_input)
        self.status_var.set("Processing your query...")
        
        # Process in background thread to keep GUI responsive
        threading.Thread(target=self.process_query, args=(user_input,)).start()
    
    def process_query(self, user_input):
        """Process user query with chatbot"""
        try:
            response = self.chatbot.generate_response(user_input)
            self.root.after(0, self.add_message, "Assistant", response)
            self.root.after(0, lambda: self.status_var.set("Ready"))
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.add_message, "Assistant", error_msg)
            self.root.after(0, lambda: self.status_var.set("Error occurred"))

# Chatbot Class with Ollama Integration
class VeterinaryChatBot:
    def __init__(self, faiss_index_dir, neo4j_uri, neo4j_user, neo4j_password, ollama_model="zephyr:7b-beta"):
        # Load embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded embedding model")
        
        # Load FAISS indices and metadata
        self.faiss_indices = {}
        self.index_metadata = {}
        for fname in os.listdir(faiss_index_dir):
            if fname.endswith(".index"):
                topic = os.path.splitext(fname)[0]
                index_path = os.path.join(faiss_index_dir, fname)
                self.faiss_indices[topic] = faiss.read_index(index_path)
                
                # Load corresponding metadata
                meta_path = os.path.join(faiss_index_dir, f"{topic}_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        self.index_metadata[topic] = json.load(f)
        print(f"Loaded {len(self.faiss_indices)} FAISS indices")
        
        # Connect to Neo4j
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"Connected to Neo4j at {neo4j_uri}")
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/chat"
        self.ollama_model = ollama_model
        print(f"Using Ollama model: {self.ollama_model}")
        
        # Entity cache for normalization
        self.entity_cache = {}
        
        # Query classification settings
        self.query_types = {
            "diagnosis": ["symptoms", "signs", "diagnose", "recognize", "differential"],
            "treatment": ["treat", "therapy", "medication", "management", "protocol"],
            "dosage": ["dose", "mg/kg", "administration", "prescription", "dosage"],
            "prevention": ["prevent", "vaccine", "biosecurity", "hygiene", "prophylaxis"],
            "etiology": ["cause", "origin", "transmission", "pathogenesis"]
        }
        
        # Preload topics and embeddings
        self.topics = list(self.faiss_indices.keys())
        if self.topics:
            self.topic_embeddings = self.embedder.encode(self.topics)
            print(f"Preloaded embeddings for {len(self.topics)} topics")
        else:
            self.topic_embeddings = np.array([])
            print("Warning: No topics found in vector store directory")
        
        # Safety parameters
        self.danger_keywords = ["euthanize", "overdose", "toxic", "contraindicated", "human-only"]
        self.dosage_pattern = re.compile(r"(\d+\s*mg/\w+)")
        print("Veterinary chatbot initialized successfully")

    def detect_query_topic(self, query):
        """Determine the most relevant topic for a query"""
        if not self.topics:
            return ""
            
        query_embedding = self.embedder.encode([query])[0]
        
        # Compute cosine similarity
        norms = np.linalg.norm(self.topic_embeddings, axis=1) * np.linalg.norm(query_embedding)
        similarities = self.topic_embeddings.dot(query_embedding) / (norms + 1e-9)
        return self.topics[np.argmax(similarities)]

    def classify_query_type(self, query):
        """Classify query type for specialized retrieval"""
        query_lower = query.lower()
        for q_type, keywords in self.query_types.items():
            if any(kw in query_lower for kw in keywords):
                return q_type
        return "general"

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context from both sources"""
        topic = self.detect_query_topic(query)
        query_type = self.classify_query_type(query)
        
        # Retrieve from vector store with hierarchy
        vector_context = self.retrieve_vector_context(query, topic, top_k)
        
        # Retrieve from knowledge graph
        entities = self.extract_entities(query)
        graph_context = self.query_knowledge_graph(entities, query_type)
        
        return vector_context + "\n\n" + graph_context

    def retrieve_vector_context(self, query, topic, top_k):
        """Retrieve context from vector store"""
        if topic not in self.faiss_indices:
            return "No relevant vector context found"
            
        query_embedding = self.embedder.encode([query])[0]
        scores, indices = self.faiss_indices[topic].search(
            np.array([query_embedding]).astype('float32'), top_k)
        
        context_lines = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.index_metadata[topic]):
                continue
                
            metadata = self.index_metadata[topic][idx]
            chunk_id = metadata.get('id', '')
            context_lines.append(self.get_chunk_hierarchy(chunk_id, i+1))
            
        if not context_lines:
            return "No vector context found"
        return "Vector Store Context:\n" + "\n".join(context_lines)

    def get_chunk_hierarchy(self, chunk_id, ref_num):
        """Retrieve document hierarchy for a chunk"""
        query = """
        MATCH (doc:Document)-[:HAS_SECTION]->(sec:Section)-[:HAS_CHUNK]->(chunk:Chunk {id: $chunk_id})
        RETURN doc.name AS document, sec.name AS section, sec.type AS section_type, chunk.text AS text
        LIMIT 1
        """
        try:
            result = self.graph.run(query, chunk_id=chunk_id).data()
            if result:
                record = result[0]
                text = record['text']
                truncated = text if len(text) <= 350 else text[:347] + '...'
                return (f"[Ref {ref_num}]: Document: {record['document']}\n"
                        f"  • Section: {record['section']} ({record['section_type']})\n"
                        f"  • Text: {truncated}")
        except Exception as e:
            print(f"Hierarchy query failed: {str(e)}")
        return f"[Ref {ref_num}]: Context not available"

    def extract_entities(self, text):
        """Enhanced veterinary entity extraction"""
        entities = []
        text_lower = text.lower()
        
        # Species detection
        species = ["cow", "dog", "cat", "horse", "sheep", "goat", "calf", 
                  "kitten", "puppy", "livestock", "poultry", "avian", "bovine", 
                  "feline", "canine", "equine", "porcine"]
        for s in species:
            if s in text_lower:
                entities.append(("SPECIES", s))
        
        # Conditions/diseases
        conditions = ["mastitis", "parvo", "rabies", "influenza", "diarrhea", 
                     "panleukopenia", "fip", "ketosis", "milk fever", "ringworm",
                     "leptospirosis", "brucellosis", "foot rot", "laminitis"]
        for c in conditions:
            if c in text_lower:
                entities.append(("CONDITION", c))
        
        # Medications
        meds = ["amoxicillin", "penicillin", "ivermectin", "doxycycline", 
               "enrofloxacin", "meloxicam", "ceftiofur", "flunixin", "oxytetracycline",
               "fenbendazole", "diazepam", "ketoprofen", "carprofen"]
        for m in meds:
            if m in text_lower:
                entities.append(("MEDICATION", m))
                
        # Body systems/anatomy
        anatomy = ["udder", "hoof", "joint", "respiratory", "gi", "reproductive",
                  "cardiovascular", "nervous", "musculoskeletal", "integumentary"]
        for a in anatomy:
            if a in text_lower:
                entities.append(("ANATOMY", a))
                
        return list(set(entities))

    def query_knowledge_graph(self, entities, query_type):
        """Retrieve context from knowledge graph using hierarchy"""
        if not entities:
            return "No relevant entities found in knowledge graph"
        
        entity_names = [e[1] for e in entities]
        query = self.get_kg_query(query_type, entity_names)
        
        try:
            result = self.graph.run(query, entity_names=entity_names).data()
            if not result:
                return "No relevant knowledge graph context found"
            
            context_lines = ["Knowledge Graph Context:"]
            for i, record in enumerate(result[:5]):
                context_lines.append(f"[KG {i+1}]: {record['info']}")
                
            return "\n".join(context_lines)
        except Exception as e:
            return f"Knowledge Graph Error: {str(e)}"

    def get_kg_query(self, query_type, entity_names):
        """Generate appropriate KG query based on query type"""
        # Medication/dosage specific handling
        if query_type == "dosage":
            return """
            MATCH (e:Entity)-[:HAS_DOSAGE|:HAS_ADVERSE_EFFECT*1..2]-(related)
            WHERE toLower(e.name) IN $entity_names
            WITH DISTINCT related
            RETURN 
              CASE 
                WHEN related:Entity THEN 'Entity: ' + related.name + ' (' + related.type + ')'
                WHEN related:Chunk THEN 'Chunk: ' + substring(related.text, 0, 250)
              END AS info
            LIMIT 5
            """
        
        # Diagnosis specific handling
        if query_type == "diagnosis":
            return """
            MATCH (e:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE toLower(e.name) IN $entity_names AND e.type = 'CONDITION'
            MATCH (chunk)<-[:HAS_CHUNK]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
            RETURN 'Document: ' + doc.name + 
                   ' | Section: ' + sec.name + 
                   ' | Condition: ' + e.name + 
                   ' | Text: ' + substring(chunk.text, 0, 200) AS info
            LIMIT 5
            """
        
        # Treatment specific handling
        if query_type == "treatment":
            return """
            MATCH (e:Entity)<-[:MENTIONS]-(chunk:Chunk)
            WHERE toLower(e.name) IN $entity_names AND e.type = 'CONDITION'
            MATCH (chunk)-[:MENTIONS]->(med:Entity {type: 'MEDICATION'})
            MATCH (chunk)<-[:HAS_CHUNK]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
            RETURN 'Document: ' + doc.name + 
                   ' | Section: ' + sec.name + 
                   ' | Condition: ' + e.name + 
                   ' | Medication: ' + med.name +
                   ' | Text: ' + substring(chunk.text, 0, 200) AS info
            LIMIT 5
            """
        
        # General query with full hierarchy
        return """
        MATCH (e:Entity)
        WHERE toLower(e.name) IN $entity_names
        MATCH (e)<-[:MENTIONS]-(chunk:Chunk)<-[:HAS_CHUNK]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
        RETURN 'Document: ' + doc.name + 
               ' | Section: ' + sec.name + ' (' + sec.type + ')' +
               ' | Entity: ' + e.name + ' (' + e.type + ')' +
               ' | Text: ' + substring(chunk.text, 0, 200) AS info
        LIMIT 5
        """

    def add_citations(self, response, context):
        """Add reference citations to the response"""
        # Extract references from context
        refs = []
        for line in context.split('\n'):
            if line.startswith('[Ref ') or line.startswith('[KG '):
                refs.append(line.split(']: ', 1)[1])
        
        if not refs:
            return response
            
        cited_response = response + "\n\nReferences:\n"
        for i, ref in enumerate(refs[:3]):
            cited_response += f"[{i+1}] {ref}\n"
        return cited_response

    def safety_check(self, response):
        """Veterinary safety validation"""
        # Check for dangerous keywords
        response_lower = response.lower()
        if any(kw in response_lower for kw in self.danger_keywords):
            response += "\n\n⚠️ WARNING: This information requires veterinary supervision"
        
        # Validate dosage formats
        dosages = self.dosage_pattern.findall(response)
        if dosages:
            response += "\n\n💊 Dosage Verification: Always confirm dosage with a veterinarian"
        
        # Add standard disclaimer
        if "consult" not in response_lower and "veterinarian" not in response_lower:
            response += "\n\nℹ️ Remember to consult a licensed veterinarian for case-specific advice"
            
        return response

    def generate_response(self, query):
        """Generate response using Ollama API"""
        try:
            # Retrieve context
            context = self.retrieve_context(query)
            
            # Format prompt for Ollama
            prompt = self.format_prompt(query, context)
            
            # Call Ollama API
            response = self.call_ollama(prompt)
            
            # Add citations and safety checks
            response = self.add_citations(response, context)
            return self.safety_check(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def call_ollama(self, prompt):
        """Call Ollama API to generate response"""
        # Correct endpoint for chat models
        data = {
            "model": self.ollama_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "No response from Ollama")
        except requests.exceptions.RequestException as e:
            # Fallback to local model if API fails
            return self.fallback_response(prompt)

    def fallback_response(self, prompt):
        """Fallback response if Ollama API fails"""
        return ("I'm having trouble connecting to the AI model. "
                "Please check that Ollama is running and try again. "
                f"\n\nYour query was: {prompt[:200]}...")

    def format_prompt(self, query, context):
        """Create veterinary-specific prompt"""
        return f"""You are a veterinary expert assistant. Follow these guidelines:
1. Provide accurate, evidence-based information using the context
2. For medication questions: Include dosage ranges, administration routes, and contraindications
3. For diagnosis questions: List key symptoms and differential diagnoses
4. For prevention questions: Recommend specific protocols and vaccination schedules
5. Always recommend professional consultation for serious cases
6. Clearly state when information is unavailable

Context:
{context}

Question: {query}

Based on the context above, please answer the veterinary question accurately and professionally.
"""

# Main execution
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = VeterinaryChatBot(
        faiss_index_dir=r"C:\Users\chidi\Documents\Chatbot vet\Vet-Knowledge-Graph-ChatBot\Scripts\vet_indexes",  # Path to your vector store
        neo4j_uri="bolt://localhost:7687",  # Your Neo4j connection URI
        neo4j_user="neo4j",  # Neo4j username
        neo4j_password="Napoleon20",  # Neo4j password
        ollama_model="zephyr"  # Use just the model name without version tag
    )
    
    # Create GUI application
    root = tk.Tk()
    app = VeterinaryChatApp(root, chatbot)
    root.mainloop()