from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent
from langchain.schema import SystemMessage
import faiss
import json
import os
import uuid
import numpy as np

class VeterinaryChatbot:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_db,
                 faiss_index_dir, model_names=["llama3:70b", "mixtral:8x22b"],
                 k=5, memory_window=10):
        # Knowledge Graph connection
        self.kg = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            database=neo4j_db
        )
        
        # FAISS indices
        self.faiss_indices = self.load_faiss_indices(faiss_index_dir)
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Ensemble LLMs
        self.llms = [Ollama(model=name) for name in model_names]
        
        # Memory and sessions
        self.sessions = {}
        self.k = k  # Top k chunks to retrieve
        self.memory_window = memory_window
        
        # Initialize tools
        self.tools = self.setup_tools()
        
        # System prompt template
        self.system_prompt = """
        You are a veterinary expert assistant. Use the following knowledge sources:
        1. Structured knowledge graph with entities and relationships
        2. Extracted document chunks with clinical information
        
        Always:
        - Cite sources using chunk IDs or entity IDs
        - Break complex questions into multi-step reasoning
        - Admit uncertainty when information is unavailable
        - Use markdown for dosage tables and treatment protocols
        """
    
    def load_faiss_indices(self, index_dir):
        """Load all FAISS indices from directory"""
        indices = {}
        for fname in os.listdir(index_dir):
            if fname.endswith(".index"):
                topic = os.path.splitext(fname)[0]
                index_path = os.path.join(index_dir, fname)
                meta_path = os.path.join(index_dir, f"{topic}_metadata.json")
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        index = faiss.read_index(index_path)
                        indices[topic] = (index, metadata)
                    except Exception as e:
                        print(f"Error loading {topic} index: {str(e)}")
        return indices
    
    def retrieve_kg_context(self, query: str) -> str:
        """Retrieve relevant subgraphs from Neo4j"""
        cypher_template = """
        CALL db.index.fulltext.queryNodes('entityIndex', $query) 
        YIELD node, score
        WITH node, score ORDER BY score DESC LIMIT 5
        MATCH (node)-[r]-(related)
        RETURN 
            node.name AS subject,
            type(r) AS relationship,
            related.name AS object,
            labels(related) AS object_type,
            score
        """
        try:
            return self.kg.query(cypher_template, params={"query": query})
        except Exception as e:
            print(f"KG query failed: {str(e)}")
            return []
    
    def retrieve_text_chunks(self, query: str) -> list:
        """Retrieve relevant text chunks from FAISS indices"""
        try:
            query_embedding = self.embedder.embed_query(query)
            results = []
            
            for topic, (index, metadata) in self.faiss_indices.items():
                D, I = index.search(np.array([query_embedding]).astype('float32'), self.k)
                for i, idx in enumerate(I[0]):
                    if idx >= 0 and idx < len(metadata):
                        chunk = metadata[idx]
                        results.append({
                            "topic": topic,
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "score": float(D[0][i])
                        })
            
            return sorted(results, key=lambda x: x["score"], reverse=True)[:self.k]
        except Exception as e:
            print(f"FAISS search failed: {str(e)}")
            return []
    
    def setup_tools(self):
        """Create tools for agent-based reasoning"""
        return [
            Tool(
                name="Knowledge_Graph",
                func=self.retrieve_kg_context,
                description="Useful for querying structured knowledge about veterinary entities and relationships"
            ),
            Tool(
                name="Document_Retrieval",
                func=self.retrieve_text_chunks,
                description="Useful for retrieving clinical text chunks from veterinary literature"
            )
        ]
    
    def create_session(self, session_id=None):
        """Create a new chat session with memory"""
        session_id = session_id or str(uuid.uuid4())
        self.sessions[session_id] = {
            "memory": ConversationBufferWindowMemory(
                k=self.memory_window,
                memory_key="chat_history",
                return_messages=True
            ),
            "thought_chain": []
        }
        return session_id
    
    def format_context(self, kg_results, chunk_results):
        """Format knowledge for LLM input"""
        context_str = "## Knowledge Graph Context:\n"
        
        if kg_results:
            context_str += "Entities and Relationships:\n"
            for record in kg_results:
                context_str += (f"- {record['subject']} --{record['relationship']}--> "
                              f"{record['object']} ({', '.join(record['object_type'])})\n")
        else:
            context_str += "No relevant KG entities found\n"
        
        context_str += "\n## Document Chunks:\n"
        if chunk_results:
            for i, chunk in enumerate(chunk_results, 1):
                context_str += (f"### Chunk {i} ({chunk['topic']}, Score: {chunk['score']:.2f})\n"
                              f"ID: {chunk['chunk_id']}\n"
                              f"Content: {chunk['text'][:300]}...\n\n")
        else:
            context_str += "No relevant text chunks found\n"
        
        return context_str
    
    def ensemble_generate(self, prompt: str) -> str:
        """Generate response using ensemble of LLMs"""
        responses = []
        for llm in self.llms:
            try:
                response = llm.invoke(prompt)
                responses.append(response)
            except Exception as e:
                print(f"Generation error with {llm.model}: {str(e)}")
        
        # Simple consensus: Select the most frequent response
        if responses:
            return max(set(responses), key=responses.count)
        return "I couldn't generate a response."

    def thought_chain_agent(self, query: str, session_id: str) -> str:
        """Multi-phase reasoning with agent-based approach"""
        session = self.sessions[session_id]
        
        # Phase 1: Knowledge retrieval
        kg_results = self.retrieve_kg_context(query)
        chunk_results = self.retrieve_text_chunks(query)
        context = self.format_context(kg_results, chunk_results)
        
        # Update thought chain
        session["thought_chain"].append({
            "phase": "retrieval",
            "query": query,
            "kg_entities": [r["subject"] for r in kg_results],
            "chunk_ids": [c["chunk_id"] for c in chunk_results]
        })
        
        # Phase 2: Agent-based reasoning
        agent_prompt = PromptTemplate.from_template(
            self.system_prompt + """
            Current Context:
            {context}
            
            Conversation History:
            {chat_history}
            
            User Question: {input}
            
            Think step-by-step and provide a comprehensive response.
            Cite sources using [Chunk ID] or [Entity] notation.
            """
        )
        
        agent = initialize_agent(
            self.tools,
            self.llms[0],  # Use primary LLM for agent
            agent="chat-conversational-react-description",
            memory=session["memory"],
            verbose=True,
            agent_kwargs={
                "system_message": SystemMessage(content=agent_prompt.template)
            }
        )
        
        response = agent.run(input=query, context=context)
        
        # Phase 3: Ensemble refinement
        refinement_prompt = f"""
        Refine the following veterinary response for accuracy and completeness: 
        
        [Original Response]
        {response}
        
        [Additional Context]
        {context}
        
        Instructions:
        1. Verify all medical facts against the context
        2. Add dosage tables where appropriate using markdown
        3. Include citations like [Chunk:123] or [Entity:Parvovirus]
        4. Maintain a professional veterinary tone
        """
        final_response = self.ensemble_generate(refinement_prompt)
        
        # Update memory and thought chain
        session["memory"].save_context(
            {"input": query},
            {"output": final_response}
        )
        session["thought_chain"].append({
            "phase": "refinement",
            "final_response": final_response
        })
        
        return final_response
    
    def chat(self, query: str, session_id: str) -> str:
        """Main chat interface"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        try:
            return self.thought_chain_agent(query, session_id)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_session_history(self, session_id: str):
        """Retrieve conversation history and thought chain"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                "history": session["memory"].load_memory_variables({})["chat_history"],
                "thought_chain": session["thought_chain"]
            }
        return None

    def list_sessions(self):
        """List all active sessions"""
        return list(self.sessions.keys())

# Initialize the chatbot
vet_bot = VeterinaryChatbot(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="Napoleon20",
    neo4j_db="vetknowledgegraphvector",
    faiss_index_dir=r"C:\Users\chidi\Documents\Chatbot vet\Vet-Knowledge-Graph-ChatBot\Scripts\vet_indexes",
    k=5,
    memory_window=8
)

# ========== USAGE EXAMPLES ========== 
# Create new chat session
session_id = vet_bot.create_session()

# Simple query
response = vet_bot.chat("What's the treatment protocol for canine parvovirus?", session_id)
print(response)

# Follow-up in same session
response = vet_bot.chat("What about vaccination schedules for puppies?", session_id)
print(response)

# Create new session for different conversation
new_session = vet_bot.create_session()
response = vet_bot.chat("Explain rumen acidosis in cattle", new_session)
print(response)

# Get session history
history = vet_bot.get_session_history(session_id)
print(json.dumps(history, indent=2))

# List all sessions
print("Active sessions:", vet_bot.list_sessions())