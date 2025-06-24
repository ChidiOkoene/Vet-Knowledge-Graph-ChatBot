import os
import json
import faiss
import numpy as np
import re
import threading
import requests
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
import spacy
from spacy.pipeline import EntityRuler

# GUI Application Class
class VeterinaryChatApp:
    def __init__(self, root, chatbot):
        self.root = root
        self.chatbot = chatbot
        self.root.title("Veterinary Expert Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f8ff')
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f8ff')
        self.style.configure('TLabel', background='#f0f8ff', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        
        header_frame = ttk.Frame(root)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        try:
            logo_img = Image.open("vet_logo.png").resize((60, 60))
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header_frame, image=self.logo)
            logo_label.pack(side=tk.LEFT, padx=5)
        except:
            pass
        
        title_label = ttk.Label(header_frame, 
                               text="Veterinary Expert Assistant", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c6fbb')
        title_label.pack(side=tk.LEFT, padx=10)
        
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
            command=self.send_message
        )
        send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.add_message("Assistant", "Hello! I'm your veterinary expert assistant. How can I help today?")
        self.user_input.focus_set()
    
    def on_enter_pressed(self, event):
        if not event.state & 0x0001:
            self.send_message()
            return "break"
        return None
    
    def add_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.tag_config('assistant', foreground='#2c6fbb', font=('Arial', 11, 'bold'))
        self.chat_display.tag_config('user', foreground='#2e8b57', font=('Arial', 11, 'bold'))
        self.chat_display.insert(tk.END, f"{sender}: ", 'assistant' if sender == "Assistant" else 'user')
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.yview(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self):
        user_input = self.user_input.get("1.0", tk.END).strip()
        if not user_input:
            return
            
        self.user_input.delete("1.0", tk.END)
        self.add_message("You", user_input)
        self.status_var.set("Processing your query...")
        threading.Thread(target=self.process_query, args=(user_input,)).start()
    
    def process_query(self, user_input):
        try:
            response = self.chatbot.generate_response(user_input)
            self.root.after(0, self.add_message, "Assistant", response)
            self.root.after(0, lambda: self.status_var.set("Ready"))
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.add_message, "Assistant", error_msg)
            self.root.after(0, lambda: self.status_var.set("Error occurred"))

# Chatbot Class with Dynamic Responses and Chunk ID References
class VeterinaryChatBot:
    def __init__(self, faiss_index_dir, neo4j_uri, neo4j_user, neo4j_password, ollama_model="zephyr:7b-beta"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded embedding model")
        
        self.faiss_indices = {}
        self.index_metadata = {}
        for fname in os.listdir(faiss_index_dir):
            if fname.endswith(".index"):
                topic = os.path.splitext(fname)[0]
                index_path = os.path.join(faiss_index_dir, fname)
                self.faiss_indices[topic] = faiss.read_index(index_path)
                
                meta_path = os.path.join(faiss_index_dir, f"{topic}_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        self.index_metadata[topic] = json.load(f)
        print(f"Loaded {len(self.faiss_indices)} FAISS indices")
        
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"Connected to Neo4j at {neo4j_uri}")
        
        self.ollama_url = "http://localhost:11434/api/chat"
        self.ollama_model = ollama_model
        print(f"Using Ollama model: {self.ollama_model}")
        
        self.entity_cache = {}
        self.entity_types = [
            "SPECIES", "DISEASE", "TREATMENT", "ANATOMY", "PROCEDURE",
            "CHEMICAL", "SYMPTOM", "PATHOGEN", "BREED", "DRUG", "DOSAGE"
        ]
        
        self.relationship_map = {
            "HAS_TREATMENT": ("DISEASE", "TREATMENT"),
            "HAS_SYMPTOM": ("DISEASE", "SYMPTOM"),
            "CAUSES": ("PATHOGEN", "DISEASE"),
            "IN": ("DISEASE", "SPECIES"),
            "HAS_DOSAGE": ("DRUG", "DOSAGE"),
            "IS_A_BREED_OF": ("BREED", "SPECIES"),
            "CAUSES_LESION_IN": ("PATHOGEN", "ANATOMY"),
            "HAS_PROCEDURE": ("CONDITION", "PROCEDURE")
        }
        
        self.topics = list(self.faiss_indices.keys())
        if self.topics:
            self.topic_embeddings = self.embedder.encode(self.topics)
            print(f"Preloaded embeddings for {len(self.topics)} topics")
        else:
            self.topic_embeddings = np.array([])
            print("Warning: No topics found")
        
        print("Enhanced veterinary chatbot initialized")
        self.nlp = self.init_nlp()
        print("Loaded NLP model with veterinary patterns")
        
    def init_nlp(self):
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        
        VETERINARY_TERMS = {
            "SYMPTOM": [
                "fever", "cough", "lameness", "vomiting", "diarrhea", "lethargy", "anorexia", "dehydration", 
                "weight loss", "jaundice", "pallor", "cyanosis", "tachycardia", "bradycardia", "tachypnea", 
                "dyspnea", "orthopnea", "apnea", "nasal discharge", "sneezing", "epistaxis", "dysphagia", 
                "regurgitation", "hematemesis", "melena", "hematochezia", "tenesmus", "constipation", 
                "abdominal distension", "bloat", "colic", "dysuria", "stranguria", "hematuria", "polyuria", 
                "oliguria", "anuria", "vaginal discharge", "testicular swelling", "mammary enlargement", 
                "paralysis", "paresis", "ataxia", "head tilt", "nystagmus", "seizures", "tremors", 
                "blindness", "deafness", "pruritus", "alopecia", "erythema", "hyperpigmentation", 
                "hypopigmentation", "papules", "pustules", "ulceration", "edema", "ascites", "cachexia", 
                "obesity", "shivering", "hypothermia", "hyperesthesia", "paresthesia", "coma", "syncope"
            ],
            "PATHOGEN": [
                "bacteria", "virus", "fungus", "parasite", "prion", "salmonella", "e.coli", "staphylococcus", 
                "aspergillus", "candida", "clostridium", "campylobacter", "leptospira", "borrelia", "brucella", 
                "mycoplasma", "chlamydia", "rickettsia", "ehrlichia", "anaplasma", "bartonella", 
                "mycobacterium", "malassezia", "microsporum", "trichophyton", "histoplasma", "blastomyces", 
                "cryptococcus", "coccidioides", "dirofilaria", "toxocara", "ancylostoma", "trichuris", 
                "dipylidium", "taenia", "echinococcus", "coccidia", "giardia", "toxoplasma", "neospora", 
                "cryptosporidium", "babesia", "cytauxzoon", "theileria", "trypanosoma", "leishmania", 
                "dermatophilus", "listeria", "yersinia", "pasteurella", "haemophilus", "actinomyces", 
                "nocardia", "erysipelothrix", "rhodococcus", "klebsiella", "proteus", "pseudomonas", 
                "francisella", "coxiella", "chlamydophila"
            ],
            "BREED": [
                "holstein", "labrador", "siamese", "thoroughbred", "yorkshire", "merino", "saanen", "leghorn", 
                "german shepherd", "golden retriever", "bulldog", "poodle", "beagle", "rottweiler", "dachshund", 
                "boxer", "siberian husky", "doberman", "australian shepherd", "corgi", "shihtzu", "pomeranian", 
                "chihuahua", "persian", "maine coon", "ragdoll", "bengal", "sphynx", "british shorthair", 
                "scottish fold", "abyssinian", "devon rex", "jersey", "angus", "hereford", "limousin", 
                "charolais", "simmental", "brahman", "dorset", "suffolk", "dorper", "nubian", "alpaca", 
                "llama", "quarter horse", "arabian", "appaloosa", "clydesdale", "shire", "pony", "duroc", 
                "hampshire", "landrace", "berkshire", "pietrain", "plymouth rock", "rhode island red", 
                "silkie", "araucana", "budgerigar", "cockatiel", "macaw", "african grey"
            ],
            "DISEASE": [
                "mastitis", "parvovirus", "brucellosis", "foot rot", "laminitis", "ketosis", "pneumonia", 
                "bronchitis", "pleuritis", "rhinitis", "gastritis", "enteritis", "colitis", "pancreatitis", 
                "hepatitis", "nephritis", "cystitis", "pyometra", "orchitis", "dermatitis", "otitis", 
                "conjunctivitis", "keratitis", "uveitis", "cardiomyopathy", "endocarditis", "anemia", 
                "leukemia", "lymphoma", "hypothyroidism", "hyperthyroidism", "diabetes", "osteomyelitis", 
                "osteochondrosis", "arthritis", "hip dysplasia", "intervertebral disc disease", "meningitis", 
                "encephalitis", "epilepsy", "cancer", "neoplasia", "abscess", "cellulitis", "sepsis", 
                "anaphylaxis", "cachexia", "obesity", "icterus", "pruritus", "alopecia", "ringworm", 
                "malassezia", "demodicosis", "sarcoptic mange", "fleas", "ticks", "heartworm", "distemper", 
                "feline leukemia", "feline immunodeficiency", "rabies"
            ],
            "TREATMENT": [
                "amoxicillin", "ivermectin", "fluid therapy", "vaccination", "antibiotic", "anthelmintic", 
                "analgesic", "enrofloxacin", "doxycycline", "cephalexin", "metronidazole", "clindamycin", 
                "gentamicin", "penicillin", "tetracycline", "ceftiofur", "flunixin", "meloxicam", "carprofen", 
                "firocoxib", "prednisolone", "dexamethasone", "insulin", "furosemide", "spironolactone", 
                "atenolol", "digoxin", "maropitant", "ondansetron", "omeprazole", "propofol", "ketamine", 
                "isoflurane", "blood transfusion", "oxygen therapy", "wound management", "bandaging", 
                "splinting", "dental prophylaxis", "chemotherapy", "radiotherapy", "acupuncture", 
                "physiotherapy", "euthanasia", "necropsy", "dietary management", "probiotics", "antifungal", 
                "antiviral", "immunosuppressant", "antihistamine", "vitamin supplementation", "mineral oil", 
                "activated charcoal", "antitoxin", "vaccine booster", "parasite prevention"
            ],
            "ANATOMY": [
                "udder", "hoof", "joint", "intestine", "rumen", "abomasum", "muscle", "tendon", "ligament", 
                "cartilage", "skin", "epidermis", "dermis", "hair follicle", "claw", "teat", "eye", "cornea", 
                "sclera", "iris", "retina", "lens", "ear", "pinna", "tympanum", "cochlea", "nose", "nostril", 
                "sinus", "mouth", "tongue", "tooth", "gum", "salivary gland", "trachea", "bronchus", 
                "alveolus", "lung", "diaphragm", "heart", "aorta", "ventricle", "atrium", "artery", "vein", 
                "liver", "gallbladder", "pancreas", "spleen", "kidney", "ureter", "bladder", "urethra", 
                "ovary", "oviduct", "uterus", "cervix", "testicle", "epididymis", "prostate", "brain", 
                "cerebrum", "cerebellum", "spinal cord", "nerve"
            ],
            "PROCEDURE": [
                "castration", "dehorning", "deworming", "biopsy", "ultrasound", "radiography", "surgery", 
                "ovariohysterectomy", "orchidectomy", "cesarean", "laparotomy", "laparoscopy", "thoracotomy", 
                "thoracoscopy", "craniotomy", "laminectomy", "amputation", "arthrotomy", "arthroscopy", 
                "osteotomy", "fracture repair", "cruciate repair", "meniscectomy", "debridement", 
                "drain placement", "endoscopy", "gastroscopy", "colonoscopy", "bronchoscopy", "cystoscopy", 
                "fluoroscopy", "computed tomography", "magnetic resonance", "electrocardiography", 
                "echocardiography", "blood pressure measurement", "urinalysis", "hematology", "biochemistry", 
                "cytology", "histopathology", "bacteriology", "mycology", "parasitology", "virology", 
                "pcr testing", "serology", "elisa", "fluid analysis", "bone marrow aspiration", 
                "cerebrospinal tap", "tracheal wash", "bronchoalveolar lavage", "skin scraping", "ear swab", 
                "fecal floatation", "urine culture", "blood culture", "fine needle aspirate"
            ],
            "CHEMICAL": [
                "chlorhexidine", "formalin", "iodine", "ethanol", "hydrogen peroxide", "sodium hypochlorite", 
                "quaternary ammonium", "phenol", "glutaraldehyde", "paracetic acid", "potassium permanganate", 
                "benzalkonium chloride", "triclosan", "chloroxylenol", "didecyl dimethyl ammonium", 
                "accelerated hydrogen peroxide", "sodium bicarbonate", "calcium hydroxide", "magnesium sulfate", 
                "calcium gluconate", "potassium chloride", "sodium chloride", "dextrose", "lactated ringers", 
                "hetastarch", "mannitol", "activated charcoal", "atropine sulfate", "dexamethasone sodium", 
                "prednisolone acetate", "insulin zinc", "heparin sodium", "warfarin", "vitamin k", "vitamin e", 
                "vitamin b complex", "vitamin c", "iron dextran", "folic acid", "cyanocobalamin", "selenium", 
                "zinc", "copper", "flunixin meglumine", "ceftiofur sodium", "enrofloxacin injectable", 
                "oxytetracycline", "erythromycin", "tilmicosin phosphate", "diazepam solution", "ketamine hcl", 
                "xylazine", "detomidine", "medetomidine", "bupivacaine", "lidocaine"
            ],
            "DRUG": [
                "amoxicillin", "enrofloxacin", "doxycycline", "cephalexin", "metronidazole", "clindamycin", 
                "marbofloxacin", "orbifloxacin", "penicillin", "tetracycline", "cefovecin", "cefpodoxime", 
                "cefquinome", "ampicillin", "tylosin", "gentamicin", "neomycin", "spectinomycin", "tilmicosin", 
                "tulathromycin", "florfenicol", "chloramphenicol", "sulfadimethoxine", "trimethoprim", 
                "nitrofurantoin", "rifampin", "ivermectin", "milbemycin", "selamectin", "moxidectin", 
                "praziquantel", "eprinomectin", "doramectin", "febantel", "febendazole", "albendazole", 
                "oxfendazole", "oxibendazole", "pyrantel", "levamisole", "morantel", "piperazine", 
                "imidacloprid", "fipronil", "meloxicam", "carprofen", "firocoxib", "grapiprant", "robenacoxib", 
                "ketoprofen", "flunixin", "phenylbutazone", "buprenorphine", "tramadol", "codeine", 
                "butorphanol", "acepromazine", "diazepam", "midazolam", "detomidine", "romifidine", 
                "medetomidine", "atipamezole"
            ],
            "DOSAGE": [
                "mg/kg", "ml/kg", "once daily", "twice daily", "three times daily", "every 8 hours", 
                "every 12 hours", "every 24 hours", "single dose", "divided doses", "intravenous", 
                "intramuscular", "subcutaneous", "oral", "topical", "ocular", "otic", "rectal", "vaginal", 
                "per os", "as needed", "with food", "on empty stomach", "loading dose", "maintenance dose", 
                "titrate to effect", "maximum daily dose", "duration of treatment", "withdrawal period", 
                "safety margin", "overdose level", "therapeutic index", "bioavailability", "half-life", 
                "peak concentration", "trough concentration", "steady state", "dose reduction", 
                "dose escalation", "weight-based dosing", "body surface area", "fixed dose", "range", 
                "minimum effective dose", "maximum tolerated dose", "dose interval", "bolus dose", 
                "continuous infusion", "tapering dose", "alternating dose", "as directed", "consult label", 
                "species-specific", "age-adjusted", "renal impairment", "hepatic impairment", "geriatric"
            ],
            "SPECIES": [
                "cow", "dog", "cat", "horse", "sheep", "goat", "pig", "chicken", "turkey", "duck", "rabbit", 
                "guinea pig", "hamster", "rat", "mouse", "ferret", "chinchilla", "bird", "parrot", "pigeon", 
                "reptile", "snake", "lizard", "turtle", "tortoise", "amphibian", "frog", "salamander", 
                "fish", "cattle", "bovine", "canine", "feline", "equine", "ovine", "caprine", "porcine", 
                "avian", "lagomorph", "rodent", "mustelid", "psittacine", "galliform", "anseriform", 
                "chelonian", "saurian", "ophidian", "camelid", "cervid", "wildlife", "zoo animal", 
                "exotic pet", "livestock", "poultry", "companion animal", "working animal", "sport animal"
            ]
        }
        
        patterns = []
        for label, terms in VETERINARY_TERMS.items():
            for term in terms:
                tokens = [{"LOWER": token.lower()} for token in term.split()]
                patterns.append({"label": label, "pattern": tokens})
        
        ruler.add_patterns(patterns)
        return nlp

    def detect_query_scope(self, query):
        query_lower = query.lower()
        if "extensive" in query_lower or "detailed" in query_lower or "all about" in query_lower:
            return "extensive"
        return "standard"
    
    def detect_query_topic(self, query):
        if not self.topics:
            return ""
            
        query_embedding = self.embedder.encode([query])[0]
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return ""
        topic_norms = np.linalg.norm(self.topic_embeddings, axis=1)
        similarities = np.dot(self.topic_embeddings, query_embedding) / (topic_norms * query_norm)
        similarities = np.nan_to_num(similarities, nan=0.0)
        return self.topics[np.argmax(similarities)]

    def classify_query_type(self, query):
        query_lower = query.lower()
        if any(term in query_lower for term in ["symptom", "sign", "condition"]):
            return "symptom"
        elif any(term in query_lower for term in ["disease", "illness", "pathology"]):
            return "disease"
        elif any(term in query_lower for term in ["treatment", "therapy", "medication"]):
            return "treatment"
        elif any(term in query_lower for term in ["anatomy", "body part", "organ"]):
            return "anatomy"
        elif any(term in query_lower for term in ["procedure", "surgery", "operation"]):
            return "procedure"
        elif any(term in query_lower for term in ["chemical", "substance", "compound"]):
            return "chemical"
        elif any(term in query_lower for term in ["breed", "species", "type"]):
            return "breed"
        elif any(term in query_lower for term in ["drug", "medication", "pharmaceutical"]):
            return "drug"
        elif any(term in query_lower for term in ["dosage", "dose", "amount"]):
            return "dosage"
        else:
            return "general"

    def retrieve_context(self, query, top_k=3):
        self.citation_map = {}
        self.citation_counter = 1
        
        scope = self.detect_query_scope(query)
        topic = self.detect_query_topic(query)
        query_type = self.classify_query_type(query)
        entities = self.extract_entities(query)
        
        context_parts = []
        context_parts.append(("DIRECT", self.retrieve_direct_context(entities, scope)))
        context_parts.append(("VECTOR", self.retrieve_vector_context(query, topic, top_k * (3 if scope == "extensive" else 1))))
        context_parts.append(("RELATIONSHIP", self.retrieve_relationship_context(entities, scope)))
        
        context_str = ""
        for context_type, context_data in context_parts:
            if not context_data or "error" in context_data.lower() or "no context" in context_data.lower():
                continue
            context_str += f"{context_type} CONTEXT:\n{context_data}\n\n"
        
        self.fill_citation_metadata()
        return context_str

    def retrieve_direct_context(self, entities, scope):
        if not entities:
            return "No direct entity context found"
        
        entity_names = [e[1].lower() for e in entities]
        limit = 10 if scope == "extensive" else 5
        
        query = """
        MATCH (e:Entity)<-[:MENTIONS]-(chunk:Chunk)
        WHERE toLower(e.name) IN $entity_names
        OPTIONAL MATCH (chunk)<-[:HAS_CHUNK]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
        RETURN 
          chunk.id AS chunk_id,
          chunk.text AS text
        ORDER BY e.type DESC
        LIMIT $limit
        """
        
        try:
            result = self.graph.run(query, entity_names=entity_names, limit=limit).data()
            if not result:
                return "No direct entity context found"
            
            context_lines = []
            for record in result:
                chunk_id = record.get('chunk_id', '')
                citation = self.add_citation_marker(chunk_id)
                text_preview = record.get('text', '')[:300] + ('...' if len(record.get('text', '')) > 300 else '')
                context_lines.append(
                    f"[Direct {citation}]: ID: {chunk_id}\n"
                    f"Text: {text_preview}"
                )
            return "\n".join(context_lines)
        except Exception as e:
            return f"Direct context error: {str(e)}"

    def retrieve_vector_context(self, query, topic, top_k):
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
            citation = self.add_citation_marker(chunk_id)
            text_preview = metadata.get('text', '')[:300] + ('...' if len(metadata.get('text', '')) > 300 else '')
            context_lines.append(
                f"[Vector {citation}]: ID: {chunk_id}\n"
                f"Text: {text_preview}"
            )
            
        if not context_lines:
            return "No vector context found"
        return "\n".join(context_lines)

    def retrieve_relationship_context(self, entities, scope):
        if not entities:
            return "No relationship context found"
        
        entity_names = [e[1].lower() for e in entities]
        limit = 15 if scope == "extensive" else 8
        
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) IN $entity_names
        MATCH (e)-[r]->(related)
        WHERE TYPE(r) IN $relationship_types
        WITH related, TYPE(r) AS rel_type
        MATCH (related)<-[:MENTIONS]-(chunk:Chunk)
        OPTIONAL MATCH (chunk)<-[:HAS_CHUNK]-(sec:Section)<-[:HAS_SECTION]-(doc:Document)
        RETURN DISTINCT
          rel_type AS relationship,
          related.name AS entity,
          related.type AS entity_type,
          chunk.id AS chunk_id,
          chunk.text AS text
        ORDER BY rel_type
        LIMIT $limit
        """
        
        try:
            result = self.graph.run(query, 
                                   entity_names=entity_names,
                                   relationship_types=list(self.relationship_map.keys()),
                                   limit=limit).data()
            if not result:
                return "No relationship context found"
            
            context_lines = []
            for record in result:
                chunk_id = record.get('chunk_id', '')
                citation = self.add_citation_marker(chunk_id)
                text_preview = record.get('text', '')[:250] + ('...' if len(record.get('text', '')) > 250 else '')
                context_lines.append(
                    f"[Rel {citation}]: {record['relationship']}: {record['entity']} [{record['entity_type']}]\n"
                    f"ID: {chunk_id}\n"
                    f"Text: {text_preview}"
                )
            return "\n".join(context_lines)
        except Exception as e:
            return f"Relationship context error: {str(e)}"

    def add_citation_marker(self, chunk_id):
        if not chunk_id:
            return ""
        if chunk_id not in self.citation_map:
            self.citation_map[chunk_id] = self.citation_counter
            self.citation_counter += 1
        return f"[{self.citation_map[chunk_id]}]"

    def fill_citation_metadata(self):
        if not self.citation_map:
            return
            
        chunk_ids = list(self.citation_map.keys())
        query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (chunk:Chunk {id: chunk_id})
        OPTIONAL MATCH (chunk)<-[:HAS_CHUNK]-(sec:Section)
        OPTIONAL MATCH (sec)<-[:HAS_SECTION]-(doc:Document)
        RETURN chunk_id, 
               COALESCE(doc.name, 'Unknown Document') AS document,
               COALESCE(sec.name, 'Unknown Section') AS section
        """
        
        try:
            result = self.graph.run(query, chunk_ids=chunk_ids).data()
            for record in result:
                chunk_id = record['chunk_id']
                if chunk_id in self.citation_map:
                    self.citation_map[chunk_id] = {
                        'num': self.citation_map[chunk_id],
                        'document': record['document'],
                        'section': record['section']
                    }
        except:
            pass

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append((ent.label_, ent.text))
        
        text_lower = text.lower()
        for label in self.entity_types:
            for term in self.get_terms_for_type(label):
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', text_lower):
                    entities.append((label, term))
        
        return list(set(entities))

    def get_terms_for_type(self, entity_type):
        return {
            "SYMPTOM": [
                "fever", "cough", "lameness", "vomiting", "diarrhea", "lethargy", "anorexia", "dehydration", 
                "weight loss", "jaundice", "pallor", "cyanosis", "tachycardia", "bradycardia", "tachypnea", 
                "dyspnea", "orthopnea", "apnea", "nasal discharge", "sneezing", "epistaxis", "dysphagia", 
                "regurgitation", "hematemesis", "melena", "hematochezia", "tenesmus", "constipation", 
                "abdominal distension", "bloat", "colic", "dysuria", "stranguria", "hematuria", "polyuria", 
                "oliguria", "anuria", "vaginal discharge", "testicular swelling", "mammary enlargement", 
                "paralysis", "paresis", "ataxia", "head tilt", "nystagmus", "seizures", "tremors", 
                "blindness", "deafness", "pruritus", "alopecia", "erythema", "hyperpigmentation", 
                "hypopigmentation", "papules", "pustules", "ulceration", "edema", "ascites", "cachexia", 
                "obesity", "shivering", "hypothermia", "hyperesthesia", "paresthesia", "coma", "syncope"
            ],
            "PATHOGEN": [
                "bacteria", "virus", "fungus", "parasite", "prion", "salmonella", "e.coli", "staphylococcus", 
                "aspergillus", "candida", "clostridium", "campylobacter", "leptospira", "borrelia", "brucella", 
                "mycoplasma", "chlamydia", "rickettsia", "ehrlichia", "anaplasma", "bartonella", 
                "mycobacterium", "malassezia", "microsporum", "trichophyton", "histoplasma", "blastomyces", 
                "cryptococcus", "coccidioides", "dirofilaria", "toxocara", "ancylostoma", "trichuris", 
                "dipylidium", "taenia", "echinococcus", "coccidia", "giardia", "toxoplasma", "neospora", 
                "cryptosporidium", "babesia", "cytauxzoon", "theileria", "trypanosoma", "leishmania", 
                "dermatophilus", "listeria", "yersinia", "pasteurella", "haemophilus", "actinomyces", 
                "nocardia", "erysipelothrix", "rhodococcus", "klebsiella", "proteus", "pseudomonas", 
                "francisella", "coxiella", "chlamydophila"
            ],
            "BREED": [
                "holstein", "labrador", "siamese", "thoroughbred", "yorkshire", "merino", "saanen", "leghorn", 
                "german shepherd", "golden retriever", "bulldog", "poodle", "beagle", "rottweiler", "dachshund", 
                "boxer", "siberian husky", "doberman", "australian shepherd", "corgi", "shihtzu", "pomeranian", 
                "chihuahua", "persian", "maine coon", "ragdoll", "bengal", "sphynx", "british shorthair", 
                "scottish fold", "abyssinian", "devon rex", "jersey", "angus", "hereford", "limousin", 
                "charolais", "simmental", "brahman", "dorset", "suffolk", "dorper", "nubian", "alpaca", 
                "llama", "quarter horse", "arabian", "appaloosa", "clydesdale", "shire", "pony", "duroc", 
                "hampshire", "landrace", "berkshire", "pietrain", "plymouth rock", "rhode island red", 
                "silkie", "araucana", "budgerigar", "cockatiel", "macaw", "african grey"
            ],
            "DISEASE": [
                "mastitis", "parvovirus", "brucellosis", "foot rot", "laminitis", "ketosis", "pneumonia", 
                "bronchitis", "pleuritis", "rhinitis", "gastritis", "enteritis", "colitis", "pancreatitis", 
                "hepatitis", "nephritis", "cystitis", "pyometra", "orchitis", "dermatitis", "otitis", 
                "conjunctivitis", "keratitis", "uveitis", "cardiomyopathy", "endocarditis", "anemia", 
                "leukemia", "lymphoma", "hypothyroidism", "hyperthyroidism", "diabetes", "osteomyelitis", 
                "osteochondrosis", "arthritis", "hip dysplasia", "intervertebral disc disease", "meningitis", 
                "encephalitis", "epilepsy", "cancer", "neoplasia", "abscess", "cellulitis", "sepsis", 
                "anaphylaxis", "cachexia", "obesity", "icterus", "pruritus", "alopecia", "ringworm", 
                "malassezia", "demodicosis", "sarcoptic mange", "fleas", "ticks", "heartworm", "distemper", 
                "feline leukemia", "feline immunodeficiency", "rabies"
            ],
            "TREATMENT": [
                "amoxicillin", "ivermectin", "fluid therapy", "vaccination", "antibiotic", "anthelmintic", 
                "analgesic", "enrofloxacin", "doxycycline", "cephalexin", "metronidazole", "clindamycin", 
                "gentamicin", "penicillin", "tetracycline", "ceftiofur", "flunixin", "meloxicam", "carprofen", 
                "firocoxib", "prednisolone", "dexamethasone", "insulin", "furosemide", "spironolactone", 
                "atenolol", "digoxin", "maropitant", "ondansetron", "omeprazole", "propofol", "ketamine", 
                "isoflurane", "blood transfusion", "oxygen therapy", "wound management", "bandaging", 
                "splinting", "dental prophylaxis", "chemotherapy", "radiotherapy", "acupuncture", 
                "physiotherapy", "euthanasia", "necropsy", "dietary management", "probiotics", "antifungal", 
                "antiviral", "immunosuppressant", "antihistamine", "vitamin supplementation", "mineral oil", 
                "activated charcoal", "antitoxin", "vaccine booster", "parasite prevention"
            ],
            "ANATOMY": [
                "udder", "hoof", "joint", "intestine", "rumen", "abomasum", "muscle", "tendon", "ligament", 
                "cartilage", "skin", "epidermis", "dermis", "hair follicle", "claw", "teat", "eye", "cornea", 
                "sclera", "iris", "retina", "lens", "ear", "pinna", "tympanum", "cochlea", "nose", "nostril", 
                "sinus", "mouth", "tongue", "tooth", "gum", "salivary gland", "trachea", "bronchus", 
                "alveolus", "lung", "diaphragm", "heart", "aorta", "ventricle", "atrium", "artery", "vein", 
                "liver", "gallbladder", "pancreas", "spleen", "kidney", "ureter", "bladder", "urethra", 
                "ovary", "oviduct", "uterus", "cervix", "testicle", "epididymis", "prostate", "brain", 
                "cerebrum", "cerebellum", "spinal cord", "nerve"
            ],
            "PROCEDURE": [
                "castration", "dehorning", "deworming", "biopsy", "ultrasound", "radiography", "surgery", 
                "ovariohysterectomy", "orchidectomy", "cesarean", "laparotomy", "laparoscopy", "thoracotomy", 
                "thoracoscopy", "craniotomy", "laminectomy", "amputation", "arthrotomy", "arthroscopy", 
                "osteotomy", "fracture repair", "cruciate repair", "meniscectomy", "debridement", 
                "drain placement", "endoscopy", "gastroscopy", "colonoscopy", "bronchoscopy", "cystoscopy", 
                "fluoroscopy", "computed tomography", "magnetic resonance", "electrocardiography", 
                "echocardiography", "blood pressure measurement", "urinalysis", "hematology", "biochemistry", 
                "cytology", "histopathology", "bacteriology", "mycology", "parasitology", "virology", 
                "pcr testing", "serology", "elisa", "fluid analysis", "bone marrow aspiration", 
                "cerebrospinal tap", "tracheal wash", "bronchoalveolar lavage", "skin scraping", "ear swab", 
                "fecal floatation", "urine culture", "blood culture", "fine needle aspirate"
            ],
            "CHEMICAL": [
                "chlorhexidine", "formalin", "iodine", "ethanol", "hydrogen peroxide", "sodium hypochlorite", 
                "quaternary ammonium", "phenol", "glutaraldehyde", "paracetic acid", "potassium permanganate", 
                "benzalkonium chloride", "triclosan", "chloroxylenol", "didecyl dimethyl ammonium", 
                "accelerated hydrogen peroxide", "sodium bicarbonate", "calcium hydroxide", "magnesium sulfate", 
                "calcium gluconate", "potassium chloride", "sodium chloride", "dextrose", "lactated ringers", 
                "hetastarch", "mannitol", "activated charcoal", "atropine sulfate", "dexamethasone sodium", 
                "prednisolone acetate", "insulin zinc", "heparin sodium", "warfarin", "vitamin k", "vitamin e", 
                "vitamin b complex", "vitamin c", "iron dextran", "folic acid", "cyanocobalamin", "selenium", 
                "zinc", "copper", "flunixin meglumine", "ceftiofur sodium", "enrofloxacin injectable", 
                "oxytetracycline", "erythromycin", "tilmicosin phosphate", "diazepam solution", "ketamine hcl", 
                "xylazine", "detomidine", "medetomidine", "bupivacaine", "lidocaine"
            ],
            "DRUG": [
                "amoxicillin", "enrofloxacin", "doxycycline", "cephalexin", "metronidazole", "clindamycin", 
                "marbofloxacin", "orbifloxacin", "penicillin", "tetracycline", "cefovecin", "cefpodoxime", 
                "cefquinome", "ampicillin", "tylosin", "gentamicin", "neomycin", "spectinomycin", "tilmicosin", 
                "tulathromycin", "florfenicol", "chloramphenicol", "sulfadimethoxine", "trimethoprim", 
                "nitrofurantoin", "rifampin", "ivermectin", "milbemycin", "selamectin", "moxidectin", 
                "praziquantel", "eprinomectin", "doramectin", "febantel", "febendazole", "albendazole", 
                "oxfendazole", "oxibendazole", "pyrantel", "levamisole", "morantel", "piperazine", 
                "imidacloprid", "fipronil", "meloxicam", "carprofen", "firocoxib", "grapiprant", "robenacoxib", 
                "ketoprofen", "flunixin", "phenylbutazone", "buprenorphine", "tramadol", "codeine", 
                "butorphanol", "acepromazine", "diazepam", "midazolam", "detomidine", "romifidine", 
                "medetomidine", "atipamezole"
            ],
            "DOSAGE": [
                "mg/kg", "ml/kg", "once daily", "twice daily", "three times daily", "every 8 hours", 
                "every 12 hours", "every 24 hours", "single dose", "divided doses", "intravenous", 
                "intramuscular", "subcutaneous", "oral", "topical", "ocular", "otic", "rectal", "vaginal", 
                "per os", "as needed", "with food", "on empty stomach", "loading dose", "maintenance dose", 
                "titrate to effect", "maximum daily dose", "duration of treatment", "withdrawal period", 
                "safety margin", "overdose level", "therapeutic index", "bioavailability", "half-life", 
                "peak concentration", "trough concentration", "steady state", "dose reduction", 
                "dose escalation", "weight-based dosing", "body surface area", "fixed dose", "range", 
                "minimum effective dose", "maximum tolerated dose", "dose interval", "bolus dose", 
                "continuous infusion", "tapering dose", "alternating dose", "as directed", "consult label", 
                "species-specific", "age-adjusted", "renal impairment", "hepatic impairment", "geriatric"
            ],
            "SPECIES": [
                "cow", "dog", "cat", "horse", "sheep", "goat", "pig", "chicken", "turkey", "duck", "rabbit", 
                "guinea pig", "hamster", "rat", "mouse", "ferret", "chinchilla", "bird", "parrot", "pigeon", 
                "reptile", "snake", "lizard", "turtle", "tortoise", "amphibian", "frog", "salamander", 
                "fish", "cattle", "bovine", "canine", "feline", "equine", "ovine", "caprine", "porcine", 
                "avian", "lagomorph", "rodent", "mustelid", "psittacine", "galliform", "anseriform", 
                "chelonian", "saurian", "ophidian", "camelid", "cervid", "wildlife", "zoo animal", 
                "exotic pet", "livestock", "poultry", "companion animal", "working animal", "sport animal"
            ]
        }.get(entity_type, [])

    def call_ollama(self, prompt):
        try:
            data = {
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Ollama API error: {str(e)}"
        
    def safety_check(self, response, query_type):
        danger_keywords = ["euthanize", "overdose", "toxic", "contraindicated"]
        response_lower = response.lower()
        
        safety_note = ""
        if any(kw in response_lower for kw in danger_keywords):
            safety_note += "\n\n⚠️ WARNING: This information requires veterinary supervision."
        
        if query_type in ["treatment", "drug"]:
            safety_note += "\n\nℹ️ DRUG SAFETY: Dosages are general guidelines. Always confirm with current formulary."
        
        if "consult" not in response_lower and "veterinarian" not in response_lower:
            safety_note += "\n\nℹ️ Remember to consult a licensed veterinarian for case-specific advice."
        
        return response + safety_note

    def generate_response(self, query):
        try:
            self.citation_map = {}
            self.citation_counter = 1
            
            context = self.retrieve_context(query)
            query_type = self.classify_query_type(query)
            prompt = self.format_dynamic_prompt(query, context, query_type)
            
            response = self.call_ollama(prompt)
            response = self.add_citations(response)
            return self.safety_check(response, query_type)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def format_dynamic_prompt(self, query, context, query_type):
        base_prompt = f"""You are a senior veterinary specialist. Use the context to answer the query.
        
Context:
{context}

Query: {query}

Structure your response:"""
        
        if query_type == "symptom":
            return base_prompt + """
[Possible Conditions]
- List 3-5 most likely diagnoses
- For each: key distinguishing features

[Recommended Actions]
- Immediate concerns to address
- Diagnostic tests to consider
- When to seek emergency care

[Supportive Care]
- Home care recommendations
- Monitoring instructions
- Warning signs to watch for"""
        
        elif query_type == "treatment":
            return base_prompt + """
[Treatment Protocol]
- First-line medications (include dosages)
- Alternative options
- Administration instructions
- Duration of treatment

[Contraindications]
- Species-specific considerations
- Drug interactions
- When to avoid

[Monitoring]
- Expected response timeline
- Adverse effects to watch for
- Follow-up schedule"""
        
        elif query_type == "disease":
            return base_prompt + """
[Disease Overview]
- Key characteristics
- Affected species
- Transmission methods

[Clinical Presentation]
- Primary symptoms
- Disease progression
- Severity indicators

[Management]
- Treatment protocols
- Prevention strategies
- Prognosis factors"""
        
        elif query_type == "drug":
            return base_prompt + """
[Drug Profile]
- Indications
- Mechanism of action
- Pharmacokinetics

[Dosage Information]
- Standard dosage ranges
- Species adjustments
- Administration routes

[Safety Considerations]
- Contraindications
- Adverse effects
- Drug interactions"""
        
        else:
            return base_prompt + """
[Comprehensive Answer]
- Provide detailed information
- Cover all relevant aspects
- Include practical recommendations

[Clinical Significance]
- Why this matters
- When to consult a vet
- Key takeaways"""

    def add_citations(self, response):
        if not self.citation_map or not isinstance(next(iter(self.citation_map.values()), {}), dict):
            return response
            
        references = "\n\nREFERENCES:\n"
        for chunk_id, info in self.citation_map.items():
            if isinstance(info, dict):  # Ensure it's the full metadata version
                references += f"[{info['num']}] Document: {info['document']}, Section: {info['section']}, Chunk ID: {chunk_id}\n"
        
        return response + references

# Main execution
if __name__ == "__main__":
    chatbot = VeterinaryChatBot(
        faiss_index_dir=r"C:\Users\chidi\Documents\Chatbot vet\Vet-Knowledge-Graph-ChatBot\Scripts\vet_indexes",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="Napoleon20",
        ollama_model="zephyr"
    )
    
    root = tk.Tk()
    app = VeterinaryChatApp(root, chatbot)
    root.mainloop()