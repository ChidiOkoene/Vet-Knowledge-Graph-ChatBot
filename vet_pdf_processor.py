#!/usr/bin/env python3

import re
import os
import uuid
import json
import fitz
import cv2
import faiss
import numpy as np
import pytesseract
import pdfplumber
import spacy
import traceback
import time
from collections import defaultdict
from PIL import Image
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
from py2neo import Graph, Node, Relationship
from spacy.pipeline import EntityRuler
from sentence_transformers import SentenceTransformer

class VeterinaryPDFProcessor:
    def __init__(self, input_dir, output_dir, faiss_index_dir, 
                 neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_database="vetknowledgegraphvector",
                 batch_size=1, max_retries=3, page_timeout=120, doc_timeout=1800):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.faiss_index_dir = faiss_index_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(faiss_index_dir, exist_ok=True)

        # Processing timeouts
        self.page_timeout = page_timeout  # Seconds per page (2 minutes)
        self.doc_timeout = doc_timeout    # Seconds per document (30 minutes)

        # Neo4j configuration
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.neo4j_driver = None

        # NLP pipeline with veterinary patterns
        self.vet_terms = self.get_veterinary_terms()
        self.nlp = self.init_nlp()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_indices = {}
        self.chunk_metadata = {}

        # Entity normalization cache
        self.entity_cache = {}
        self.processed_docs = set()
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Enhanced section detection
        self.section_detectors = [
            self.detect_section_from_toc,
            self.detect_section_from_headings,
            self.detect_section_from_content
        ]

        # Initialize Neo4j connection
        if neo4j_uri and neo4j_user and neo4j_password:
            self.neo4j_driver = self.connect_neo4j_with_retry(
                neo4j_uri, neo4j_user, neo4j_password, max_retries
            )

    # Extensive veterinary dictionary
    def get_veterinary_terms(self):
        return [
        # Common Drugs & Medications (150+ terms)
        "amoxicillin", "enrofloxacin", "doxycycline", "cephalexin", "metronidazole",
        "clindamycin", "marbofloxacin", "orbifloxacin", "penicillin", "tetracycline",
        "cefovecin", "cefpodoxime", "cefquinome", "ampicillin", "tylosin", "gentamicin",
        "neomycin", "spectinomycin", "tilmicosin", "tulathromycin", "florfenicol",
        "chloramphenicol", "sulfadimethoxine", "trimethoprim", "nitrofurantoin", "rifampin",
        "ivermectin", "milbemycin", "selamectin", "moxidectin", "praziquantel", "eprinomectin",
        "doramectin", "febantel", "febendazole", "albendazole", "oxfendazole", "oxibendazole",
        "pyrantel", "levamisole", "morantel", "piperazine", "imidacloprid", "fipronil",
        "meloxicam", "carprofen", "firocoxib", "grapiprant", "robenacoxib", "ketoprofen",
        "flunixin", "phenylbutazone", "aspirin", "acetaminophen", "codeine", "tramadol",
        "buprenorphine", "morphine", "fentanyl", "butorphanol", "hydromorphone", "methadone",
        "prednisolone", "dexamethasone", "triamcinolone", "methylprednisolone", "flumethasone",
        "acepromazine", "diazepam", "midazolam", "xylazine", "detomidine", "romifidine",
        "medetomidine", "atipamezole", "atropine", "glycopyrrolate", "epinephrine", "dopamine",
        "dobutamine", "furosemide", "spironolactone", "benazepril", "enalapril", "pimobendan",
        "atenolol", "propranolol", "diltiazem", "amlodipine", "digoxin", "insulin", "glargine",
        "glipizide", "levothyroxine", "methimazole", "deslorelin", "leuprolide", "maropitant",
        "ondansetron", "metoclopramide", "cisapride", "famotidine", "omeprazole", "sucralfate",
        "propofol", "ketamine", "alfaxalone", "isoflurane", "sevoflurane", "halothane",
        "etomidate", "thiopental", "guaifenesin", "lidocaine", "bupivacaine", "propantheline",
        "aminophylline", "theophylline", "cyproheptadine", "apomorphine", "activated charcoal",

        # Pathogens & Parasites (150+ terms)
        "parvovirus", "distemper", "adenovirus", "herpesvirus", "calicivirus", "coronavirus",
        "rabies", "influenza", "parainfluenza", "leptospira", "borrelia", "brucella", "salmonella",
        "campylobacter", "escherichia", "clostridium", "staphylococcus", "streptococcus", "pseudomonas",
        "klebsiella", "proteus", "mycoplasma", "chlamydia", "rickettsia", "ehrlichia", "anaplasma",
        "bartonella", "mycobacterium", "aspergillus", "malassezia", "microsporum", "trichophyton",
        "candida", "histoplasma", "blastomyces", "cryptococcus", "coccidioides", "dirofilaria",
        "toxocara", "ancylostoma", "trichuris", "dipylidium", "taenia", "echinococcus", "mesocestoides",
        "spirometra", "coccidia", "giardia", "toxoplasma", "neospora", "cryptosporidium", "sarcocystis",
        "babesia", "cytauxzoon", "theileria", "trypanosoma", "leishmania", "hepaticola", "capillaria",
        "aelurostrongylus", "angiostrongylus", "dracunculus", "dioctophyma", "habronema", "onchocerca",
        "setaria", "stephanurus", "strongylus", "trichostrongylus", "dictyocaulus", "protostrongylus",
        "muellerius", "metastrongylus", "parascaris", "strongyloides", "oxyuris", "syphacia", "aspiculuris",
        "psoroptes", "sarcoptes", "demodex", "cheyletiella", "otodectes", "chorioptes", "linognathus",
        "haematopinus", "solenspora", "polyplax", "ctenocephalides", "echidnophaga", "pulex", "xenopsylla",
        "amblyomma", "dermacentor", "rhipicephalus", "haemaphysalis", "ixodes", "ornithodoros", "otobius",
        "trombicula", "neotrombicula", "liponyssoides", "cheyletus", "tyrophagus", "acarus", "psorergates",

        # Diseases & Conditions (150+ terms)
        "mastitis", "ketosis", "laminitis", "pneumonia", "bronchitis", "pleuritis", "tracheitis",
        "rhinitis", "sinusitis", "pharyngitis", "tonsillitis", "gastritis", "enteritis", "colitis",
        "pancreatitis", "hepatitis", "cholangitis", "cholecystitis", "peritonitis", "nephritis",
        "pyelonephritis", "cystitis", "urethritis", "prostatitis", "balanoposthitis", "vaginitis",
        "endometritis", "pyometra", "metritis", "orchitis", "epididymitis", "dermatitis", "pyoderma",
        "pododermatitis", "otitis", "conjunctivitis", "keratitis", "uveitis", "endophthalmitis",
        "glaucoma", "cataract", "retinopathy", "cardiomyopathy", "endocarditis", "pericarditis",
        "myocarditis", "arrhythmia", "anemia", "leukemia", "lymphoma", "thrombocytopenia", "coagulopathy",
        "hypothyroidism", "hyperthyroidism", "hypoadrenocorticism", "hyperadrenocorticism", "diabetes",
        "insulinoma", "hypoglycemia", "osteomyelitis", "osteochondrosis", "osteoporosis", "osteosarcoma",
        "arthritis", "polyarthritis", "spondylosis", "hip dysplasia", "elbow dysplasia", "patellar luxation",
        "cruciate rupture", "meniscal tear", "intervertebral disc disease", "spondylitis", "meningitis",
        "encephalitis", "myelitis", "radiculitis", "polyneuropathy", "myopathy", "seizures", "ataxia",
        "tremor", "nystagmus", "head tilt", "paralysis", "paresis", "cancer", "neoplasia", "adenocarcinoma",
        "carcinoma", "sarcoma", "hemangiosarcoma", "lymphosarcoma", "mast cell tumor", "melanoma",
        "histiocytoma", "fibrosarcoma", "osteochondroma", "chondrosarcoma", "lipoma", "hematoma",
        "abscess", "cellulitis", "gangrene", "necrosis", "edema", "ascites", "cachexia", "obesity",
        "anorexia", "polyphagia", "polydipsia", "polyuria", "oliguria", "anuria", "dysuria", "hematuria",
        "stranguria", "tenesmus", "diarrhea", "constipation", "vomiting", "regurgitation", "dysphagia",
        "pica", "icterus", "cyanosis", "pallor", "petechiae", "ecchymosis", "pruritus", "alopecia",
        "erythema", "hyperkeratosis", "lichenification", "ulceration", "fistula", "granuloma",

        # Anatomy & Physiology (100+ terms)
        "abomasum", "rumen", "reticulum", "omasum", "intestine", "duodenum", "jejunum", "ileum",
        "colon", "cecum", "rectum", "anus", "pancreas", "liver", "gallbladder", "spleen", "kidney",
        "ureter", "bladder", "urethra", "ovary", "oviduct", "uterus", "cervix", "vagina", "vulva",
        "testicle", "epididymis", "vas deferens", "prostate", "bulbourethral", "seminal vesicle",
        "heart", "aorta", "ventricle", "atrium", "valve", "artery", "vein", "capillary", "trachea",
        "bronchus", "bronchiole", "alveolus", "lung", "diaphragm", "pleura", "thyroid", "parathyroid",
        "adrenal", "pituitary", "hypothalamus", "pineal", "cerebrum", "cerebellum", "brainstem",
        "spinal cord", "nerve", "ganglion", "vertebra", "femur", "tibia", "fibula", "humerus", "radius",
        "ulna", "scapula", "pelvis", "patella", "tarsus", "carpus", "phalange", "tendon", "ligament",
        "cartilage", "synovium", "meniscus", "bursa", "muscle", "fascia", "skin", "epidermis", "dermis",
        "hypodermis", "hair", "follicle", "claw", "hoof", "udder", "teat", "eye", "cornea", "sclera",
        "iris", "retina", "lens", "ear", "pinna", "tympanum", "cochlea", "vestibule", "nose", "nostril",
        "sinus", "mouth", "tongue", "tooth", "gum", "salivary gland",

        # Procedures & Diagnostics (100+ terms)
        "ovariohysterectomy", "castration", "orchidectomy", "cesarean", "laparotomy", "laparoscopy",
        "thoracotomy", "thoracoscopy", "craniotomy", "laminectomy", "amputation", "arthrotomy", "arthroscopy",
        "osteotomy", "fracture repair", "cruciate repair", "meniscectomy", "debridement", "drain placement",
        "biopsy", "endoscopy", "gastroscopy", "colonoscopy", "bronchoscopy", "cystoscopy", "ultrasonography",
        "radiography", "fluoroscopy", "computed tomography", "magnetic resonance", "electrocardiography",
        "echocardiography", "blood pressure", "urinalysis", "hematology", "biochemistry", "cytology",
        "histopathology", "bacteriology", "mycology", "parasitology", "virology", "pcr", "serology",
        "elisa", "fluid therapy", "transfusion", "oxygen therapy", "nebulization", "bandaging", "splinting",
        "casting", "wound management", "debridement", "skin graft", "dental prophylaxis", "scaling", "polishing",
        "extraction", "root planing", "periodontal therapy", "endodontics", "restoration", "prosthodontics",
        "vaccination", "microchipping", "euthanasia", "necropsy", "auscultation", "palpation", "percussion",
        "thermography", "electromyography", "arthrocentesis", "abdominocentesis", "thoracocentesis",
        "cystocentesis", "bone marrow", "cerebrospinal", "tracheal wash", "bronchoalveolar", "transtracheal",

        # Species & Breeds (100+ terms)
        "canine", "feline", "bovine", "ovine", "caprine", "equine", "porcine", "avian", "lapine", "cavia",
        "psittacine", "galliform", "anseriform", "columbiform", "passerine", "reptile", "chelonian", "saurian",
        "ophidian", "amphibian", "rodent", "mustelid", "marsupial", "primate", "cetacean", "pinniped", "ursid",
        "labrador", "german shepherd", "golden retriever", "bulldog", "poodle", "beagle", "rottweiler", "dachshund",
        "boxer", "siberian husky", "doberman", "australian shepherd", "corgi", "shihtzu", "pomeranian", "chihuahua",
        "siamese", "persian", "maine coon", "ragdoll", "bengal", "sphynx", "british shorthair", "scottish fold",
        "abyssinian", "devon rex", "holstein", "jersey", "angus", "hereford", "limousin", "charolais", "simmental",
        "brahman", "merino", "dorset", "suffolk", "dorper", "boer", "saanen", "nubian", "alpaca", "llama",
        "thoroughbred", "quarter horse", "arabian", "appaloosa", "clydesdale", "shire", "pony", "yorkshire",
        "duroc", "hampshire", "landrace", "berkshire", "pietrain", "leghorn", "plymouth rock", "rhode island red",
        "silkie", "araucana", "budgerigar", "cockatiel", "macaw", "african grey", "amazon", "cockatoo", "finch",
        "canary", "zebra finch", "iguana", "bearded dragon", "gecko", "chameleon", "tortoise", "turtle", "python",
        "boa", "kingsnake", "frog", "salamander", "axolotl", "hedgehog", "ferret", "chinchilla", "gerbil", "hamster",

        # Chemicals & Disinfectants (50+ terms)
        "chlorhexidine", "povidone iodine", "ethanol", "isopropanol", "hydrogen peroxide", "sodium hypochlorite",
        "quaternary ammonium", "phenol", "glutaraldehyde", "formaldehyde", "paracetic acid", "potassium permanganate",
        "benzalkonium chloride", "triclosan", "chloroxylenol", "didecyl dimethyl ammonium", "accelerated hydrogen peroxide",
        "sodium bicarbonate", "calcium hydroxide", "magnesium sulfate", "calcium gluconate", "potassium chloride",
        "sodium chloride", "dextrose", "lactated ringers", "hetastarch", "mannitol", "activated charcoal", "atropine sulfate",
        "dexamethasone sodium", "prednisolone acetate", "insulin zinc", "heparin sodium", "warfarin", "vitamin k",
        "vitamin e", "vitamin b complex", "vitamin c", "iron dextran", "folic acid", "cyanocobalamin", "selenium",
        "zinc", "copper", "iodine", "flunixin meglumine", "ceftiofur sodium", "enrofloxacin injectable", "oxytetracycline",
        "erythromycin", "tilmicosin phosphate",

        # Additional Veterinary Terms (100+ terms)
        "zoonosis", "nosocomial", "iatrogenic", "idiopathic", "congenital", "hereditary", "neoplastic", "inflammatory",
        "degenerative", "metabolic", "toxic", "nutritional", "autoimmune", "allergic", "infectious", "contagious",
        "endemic", "epidemic", "pandemic", "acute", "chronic", "subacute", "peracute", "anaphylaxis", "sepsis", "septicemia",
        "bacteremia", "viremia", "toxemia", "shock", "dehydration", "malnutrition", "starvation", "obesity", "cachexia",
        "fever", "pyrexia", "hyperthermia", "hypothermia", "tachycardia", "bradycardia", "tachypnea", "bradypnea", "dyspnea",
        "apnea", "hyperpnea", "hypopnea", "orthopnea", "cyanosis", "pallor", "icterus", "jaundice", "petechiae", "ecchymosis",
        "purpura", "pruritus", "itch", "alopecia", "hypotrichosis", "erythema", "hyperemia", "pustule", "papule", "vesicle",
        "bullae", "wheal", "scale", "crust", "excoriation", "lichenification", "hyperpigmentation", "hypopigmentation",
        "ulceration", "erosion", "fissure", "sinus", "fistula", "necrosis", "gangrene", "edema", "anasarca", "ascites",
        "effusion", "transudate", "exudate", "anorexia", "inappetence", "polyphagia", "polydipsia", "polyuria", "oliguria",
        "anuria", "dysuria", "stranguria", "hematuria", "proteinuria", "pyuria", "crystalluria", "cylindruria", "diarrhea",
        "dysentery", "melena", "hematochezia", "constipation", "obstipation", "tenesmus", "vomiting", "emesis", "regurgitation",
        "dysphagia", "odynophagia", "pica", "bloat", "tympany", "ataxia", "incoordination", "paresis", "paralysis", "plegia",
        "monoparesis", "paraparesis", "tetraparesis", "hemiparesis", "tremor", "fasciculation", "seizure", "convulsion",
        "nystagmus", "strabismus", "miosis", "mydriasis", "anisocoria", "blindness", "amaurosis", "deafness", "paresthesia",
        "neuralgia", "coma", "stupor", "lethargy", "depression", "hyperesthesia", "anesthesia", "analgesia", "sedation"
    ]

    def connect_neo4j_with_retry(self, uri, user, password, max_retries, delay=5):
        for attempt in range(max_retries):
            try:
                # Connect and specify the target database
                driver = Graph(uri, auth=(user, password), name=self.neo4j_database)

                # Simple connectivity test with py2neo
                cursor = driver.run("RETURN 'Connection successful' AS message, 1 AS status")
                records = cursor.data()         # returns a list of dicts
                record = records[0] if records else {"message": "", "status": 0}

                print(f"✅ Connected to Neo4j. Status: {record['status']}")
                print(f"📊 Using database: {self.neo4j_database}")
                return driver

            except Exception as e:
                print(f"⚠️ Connection attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = delay * (2 ** attempt)
                    print(f"⏳ Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        raise ConnectionError(f"Failed to connect to Neo4j after {max_retries} attempts")



    def ensure_neo4j_connection(self):
        """Ensure we have an active connection before operations"""
        if not self.neo4j_driver:
            return False

        try:
            # Simple connection test
            self.neo4j_driver.run("RETURN 1")
            return True
        except Exception as e:
            print(f"Neo4j connection lost: {str(e)}. Attempting to reconnect...")
            try:
                self.neo4j_driver = self.connect_neo4j_with_retry(
                    self.neo4j_uri, self.neo4j_user, self.neo4j_password, self.max_retries
                )
                return True
            except Exception as e2:
                print(f"Reconnection failed: {str(e2)}")
                return False

    def init_nlp(self):
        """Initialize spaCy NLP pipeline with veterinary patterns"""
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        ruler = nlp.add_pipe("entity_ruler", before="ner")

        patterns = []
        for term in self.vet_terms:
            patterns.append({"label": "TERM", "pattern": [{"LOWER": term.lower()}]})

        patterns.extend([
            {"label": "SPECIES", "pattern": [{"LOWER": {"IN": [
                "cow", "bovine", "goat", "caprine", "dog", "canine", 
                "cat", "feline", "chicken", "avian", "sheep", "ovine",
                "pig", "porcine", "horse", "equine"
            ]}}]},
            {"label": "DISEASE", "pattern": [{"LOWER": {"IN": [
                "mastitis", "parvovirus", "brucellosis", "foot rot",
                "ringworm", "laminitis", "ketosis", "pneumonia"
            ]}}]},
            {"label": "TREATMENT", "pattern": [{"LOWER": {"IN": [
                "amoxicillin", "ivermectin", "fluid therapy", "vaccination",
                "antibiotic", "anthelmintic", "analgesic"
            ]}}]},
            {"label": "ANATOMY", "pattern": [{"LOWER": {"IN": [
                "udder", "hoof", "joint", "intestine", "rumen", "abomasum",
                "muscle", "tendon", "ligament"
            ]}}]},
            {"label": "PROCEDURE", "pattern": [{"LOWER": {"IN": [
                "castration", "dehorning", "deworming", "biopsy",
                "ultrasound", "radiography", "surgery"
            ]}}]},
            {"label": "CHEMICAL", "pattern": [{"LOWER": {"IN": [
                "chlorhexidine", "formalin", "iodine", "ethanol",
                "hydrogen peroxide", "sodium hypochlorite"
            ]}}]}
        ])
        ruler.add_patterns(patterns)
        return nlp

    # --------------------- TOC-BASED SECTION DETECTION ---------------------
    def extract_toc(self, pdf_path):
        toc = []

        # Method 1: Extract using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if hasattr(pdf, 'outline') and pdf.outline:
                    toc = self.parse_pdfplumber_outline(pdf.outline)
        except:
            pass

        # Method 2: Extract using PyMuPDF
        if not toc:
            try:
                with fitz.open(pdf_path) as doc:
                    toc = self.parse_fitz_toc(doc.get_toc())
            except:
                pass

        # Method 3: Find ToC pages by content
        if not toc:
            try:
                toc = self.find_toc_by_content(pdf_path)
            except:
                pass

        return toc

    def parse_pdfplumber_outline(self, outline, level=0):
        toc = []
        for item in outline:
            if isinstance(item, dict):
                title = item.get('title', '').strip()
                page = item.get('page', 0) + 1
                if title and page > 0:
                    toc.append({'title': title, 'page': page, 'level': level})
                if 'children' in item:
                    toc.extend(self.parse_pdfplumber_outline(item['children'], level+1))
        return toc

    def parse_fitz_toc(self, toc):
        parsed = []
        for item in toc:
            if len(item) >= 3:
                level, title, page = item[:3]
                parsed.append({'title': title.strip(), 'page': page + 1, 'level': level})
        return parsed

    def find_toc_by_content(self, pdf_path):
        toc_candidates = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue

                toc_score = 0
                if re.search(r'\btable\s+of\s+contents?\b', text, re.I):
                    toc_score += 3
                if re.search(r'\bcontents?\b', text, re.I):
                    toc_score += 2
                if re.search(r'\bchapter\b|\bsection\b|\bpage\b', text, re.I):
                    toc_score += 1

                lines = text.split('\n')
                toc_lines = 0
                for line in lines:
                    if re.match(r'^(chapter|section|part)\s+\w+\s+\.+\s+\d+$', line, re.I):
                        toc_lines += 1
                    elif re.match(r'^\d+(\.\d+)*\s+.+\.+\s+\d+$', line):
                        toc_lines += 1

                if toc_lines > 3:
                    toc_score += toc_lines / 2

                if toc_score >= 3:
                    toc_candidates.append((page_num, toc_score))

        for page_num, score in sorted(toc_candidates, key=lambda x: x[1], reverse=True):
            toc = self.extract_toc_from_page(pdf_path, page_num)
            if toc:
                return toc

        return []

    def extract_toc_from_page(self, pdf_path, page_num):
        toc = []
        with pdfplumber.open(pdf_path) as pdf:
            try:
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                lines = text.split('\n')

                for line in lines:
                    match = re.match(r'^(chapter|section|part)\s*(\d+(\.\d+)*)[:\.]?\s*(.+?)\s*\.+\s*(\d+)$', line, re.I)
                    if match:
                        toc.append({
                            'title': f"{match.group(1).title()} {match.group(2)}: {match.group(4)}",
                            'page': int(match.group(5)),
                            'level': 1 if 'chapter' in match.group(1).lower() else 2
                        })
                        continue

                    match = re.match(r'^(\d+(\.\d+)*)\s+(.+?)\s*\.+\s*(\d+)$', line)
                    if match:
                        level = match.group(1).count('.') + 1
                        toc.append({
                            'title': match.group(3),
                            'page': int(match.group(4)),
                            'level': level
                        })
                        continue

                    match = re.match(r'^(.+?)\s*\.+\s*(\d+)$', line)
                    if match:
                        toc.append({
                            'title': match.group(1),
                            'page': int(match.group(2)),
                            'level': 2
                        })
            except:
                pass

        return toc

    def build_section_map(self, toc):
        section_map = {}
        if not toc:
            return section_map

        toc.sort(key=lambda x: x['page'])

        for i in range(len(toc)):
            start_page = toc[i]['page']
            end_page = toc[i+1]['page'] - 1 if i < len(toc) - 1 else float('inf')
            section_map[start_page] = {'title': toc[i]['title'], 'end_page': end_page}

        return section_map

    def detect_section_from_toc(self, section_map, page_num):
        candidate = None
        for start_page, section_data in sorted(section_map.items()):
            if start_page <= page_num <= section_data['end_page']:
                candidate = section_data['title']
                break
        return candidate or "CONTENT"

    def detect_section_from_headings(self, text):
        lines = text.split('\n')
        for line in lines[:5]:
            if 10 < len(line) < 100:
                return line.strip()
        return "CONTENT"

    def detect_section_from_content(self, text):
        patterns = [
            (r'\b(introduction|background)\b', "INTRODUCTION"),
            (r'\b(materials? & methods|procedure)\b', "METHODS"),
            (r'\b(results?|findings)\b', "RESULTS"),
            (r'\b(discussion|analysis)\b', "DISCUSSION"),
            (r'\b(conclusion|recommendations)\b', "CONCLUSION"),
            (r'\b(references?|bibliography)\b', "REFERENCES")
        ]

        for pattern, section_type in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return section_type
        return "CONTENT"

    # --------------------- CORE PROCESSING METHODS ---------------------
    def safe_get_pixmap(self, page, zoom=2, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            except Exception as e:
                if attempt == max_attempts - 1:
                    width = int(page.rect.width * zoom)
                    height = int(page.rect.height * zoom)
                    return fitz.Pixmap(fitz.csRGB, width, height, (255, 255, 255))
                else:
                    time.sleep(0.5 * (attempt + 1))

    def is_mostly_blank(self, img, threshold=0.95):
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            gray = img.convert('L')
            np_array = np.array(gray)
            non_white = np_array < 240
            non_white_ratio = np.mean(non_white)
            return non_white_ratio < (1 - threshold)
        except:
            return False

    def preprocess_image_for_ocr(self, img):
        try:
            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            return Image.fromarray(dilated)
        except:
            return img

    def extract_text(self, pdf_path):
        texts = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if len(page_text.strip()) < 100:
                            page_text = self.ocr_page(pdf_path, i)
                        texts.append(page_text)
                    except:
                        texts.append(self.ocr_page(pdf_path, i))
            return texts
        except:
            return self.extract_text_fallback(pdf_path)

    def extract_text_fallback(self, pdf_path):
        texts = []
        try:
            with fitz.open(pdf_path) as doc:
                for i in range(len(doc)):
                    page = doc.load_page(i)
                    text = page.get_text()
                    if not text or len(text.strip()) < 100:
                        text = self.ocr_page(pdf_path, i)
                    texts.append(text)
            return texts
        except:
            return [""]

    def ocr_page(self, pdf_path, page_index):
        try:
            start_time = time.time()
            with fitz.open(pdf_path) as doc:
                page = doc.load_page(page_index)
                pix = self.safe_get_pixmap(page)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                if self.is_mostly_blank(img):
                    return ""

                processed_img = self.preprocess_image_for_ocr(img)

                dict_path = f"vet_terms_{uuid.uuid4().hex[:8]}.txt"
                with open(dict_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.vet_terms))

                custom_config = (
                    r'--oem 3 --psm 6 '
                    r'-c preserve_interword_spaces=1 '
                    f'--user-words {dict_path}'
                )

                try:
                    text = pytesseract.image_to_string(processed_img, config=custom_config, timeout=self.page_timeout) or ""
                except RuntimeError:
                    text = ""

                try:
                    os.remove(dict_path)
                except:
                    pass

                return text
        except:
            return ""

    def clean_text(self, text):
        replacements = {
            r'\b(\w+)\s*-\s*\n\s*(\w+)\b': r'\1\2',
            r'\b(\w)\s+(\w)\b': r'\1\2',
            r'\b(\w+)\s*\\\s*(\w+)\b': r'\1\2',
            r'\b(\w{1,2})\s+(\w{2,})\b': r'\1\2'
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r'\bPage\s*\d+\b', '', text)
        text = re.sub(r'\.{2,}', ' ', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'-\n+', '', text)
        return text.strip()

    def segment_into_chunks(self, text, max_chars=1500):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= max_chars:
                current_chunk += ' ' + sent if current_chunk else sent
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def extract_entities(self, text):
        text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if re.match(r'^\w[\w\s-]*\w$', ent.text):
                entities.append((ent.text, ent.label_))

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text
            if len(chunk_text.split()) > 0 and re.match(r'^\w[\w\s-]*\w$', chunk_text):
                entities.append((chunk_text, "CONCEPT"))

        seen = set()
        return [(text, label) for text, label in entities 
                if not (text.lower(), label) in seen and not seen.add((text.lower(), label))]

    def store_embeddings(self, chunk, topic, chunk_id):
        embedding = self.embedder.encode([chunk])[0]

        if topic not in self.faiss_indices:
            index = faiss.IndexFlatIP(embedding.shape[0])
            self.faiss_indices[topic] = index
            self.chunk_metadata[topic] = []

        index = self.faiss_indices[topic]
        index.add(np.array([embedding]).astype('float32'))
        self.chunk_metadata[topic].append({
            "chunk_id": chunk_id,
            "text": chunk[:500] + "..." if len(chunk) > 500 else chunk
        })

    def create_neo4j_hierarchy(self, doc_name, sections):
        """Create hierarchy with py2neo v4 compatibility"""
        if not self.ensure_neo4j_connection():
            print("Skipping Neo4j export due to connection issues")
            return 0

        triple_count = 0

        try:
            tx = self.neo4j_driver.begin()

            # Create document node
            doc_node = Node("Document", name=doc_name, id=f"DOC_{uuid.uuid4().hex[:10]}")
            tx.merge(doc_node, "Document", "id")
            triple_count += 1

            for section_data in sections:
                section_node = Node("Section", 
                                name=section_data['name'],
                                type=section_data['type'],
                                page=section_data['page'],
                                id=f"SECT_{uuid.uuid4().hex[:8]}")
                tx.merge(section_node, "Section", "id")
                triple_count += 1

                tx.create(Relationship(doc_node, "HAS_SECTION", section_node))
                triple_count += 1

                for chunk_data in section_data['chunks']:
                    chunk_node = Node("Chunk", 
                                    id=chunk_data['id'],
                                    text=chunk_data['text'],
                                    topic=chunk_data['topic'])
                    tx.create(chunk_node)
                    triple_count += 1

                    tx.create(Relationship(section_node, "HAS_CHUNK", chunk_node))
                    triple_count += 1

                    for entity_text, entity_label in chunk_data['entities']:
                        entity_id = self.normalize_entity(entity_text, entity_label)
                        entity_node = Node("Entity", 
                                        id=entity_id,
                                        name=entity_text,
                                        type=entity_label)
                        tx.merge(entity_node, "Entity", "id")

                        if entity_id not in self.entity_cache.values():
                            triple_count += 1

                        tx.create(Relationship(chunk_node, "MENTIONS", entity_node))
                        triple_count += 1

            tx.commit()
            return triple_count
        except Exception as e:
            print(f"Neo4j transaction failed: {str(e)}")
            traceback.print_exc()
            if tx: tx.rollback()
            return 0

    def normalize_entity(self, text, label):
        key = (text.lower(), label)
        if key in self.entity_cache:
            return self.entity_cache[key]

        for (cache_text, cache_label), eid in self.entity_cache.items():
            if cache_label == label and fuzz.ratio(cache_text, text.lower()) > 85:
                return eid

        entity_id = f"{label}_{uuid.uuid4().hex[:8]}"
        self.entity_cache[key] = entity_id
        return entity_id

    def sanitize_filename(self, name):
        return re.sub(r'[^\w\-]', '_', name)

    def process_document(self, filename):
        if filename in self.processed_docs:
            return {"status": "skipped", "reason": "already processed"}

        self.processed_docs.add(filename)
        pdf_path = os.path.join(self.input_dir, filename)
        doc_name = os.path.splitext(filename)[0]
        sanitized_doc_name = self.sanitize_filename(doc_name)

        doc_stats = {
            "pages": 0,
            "chunks": 0,
            "entities": 0,
            "triples": 0,
            "status": "processed"
        }

        doc_start_time = time.time()

        try:
            toc = self.extract_toc(pdf_path)
            section_map = self.build_section_map(toc)
            pages = self.extract_text(pdf_path)
            doc_stats["pages"] = len(pages)
            print(f"Processing {filename} with {len(pages)} pages")
        except Exception as e:
            return {"status": "failed", "error": f"Initial processing failed: {str(e)}"}

        document_sections = []

        for page_num, page_text in enumerate(pages, 1):
            page_start_time = time.time()

            if page_num % 10 == 0 or page_num == len(pages):
                print(f"  Processed page {page_num}/{len(pages)}")

            try:
                clean_text = self.clean_text(page_text)

                section_title = "CONTENT"
                for detector in self.section_detectors:
                    try:
                        if detector == self.detect_section_from_toc:
                            section_title = detector(section_map, page_num)
                        else:
                            section_title = detector(clean_text)

                        if section_title != "CONTENT":
                            break
                    except:
                        continue

                chunks = self.segment_into_chunks(clean_text)
                section_data = {
                    "name": section_title,
                    "type": "SECTION",
                    "page": page_num,
                    "chunks": []
                }

                for chunk_idx, chunk_text in enumerate(chunks, 1):
                    try:
                        entities = self.extract_entities(chunk_text)
                        doc_stats["entities"] += len(entities)
                    except:
                        entities = []

                    chunk_topic = "GENERAL"
                    if any(e[1] == "DISEASE" for e in entities):
                        chunk_topic = "DISEASE"
                    elif any(e[1] == "TREATMENT" for e in entities):
                        chunk_topic = "TREATMENT"
                    elif any(e[1] == "ANATOMY" for e in entities):
                        chunk_topic = "ANATOMY"
                    elif any(e[1] == "PROCEDURE" for e in entities):
                        chunk_topic = "PROCEDURE"

                    chunk_id = f"{sanitized_doc_name}_page{page_num}_chunk{chunk_idx}"
                    chunk_filename = f"{chunk_id}.txt"

                    try:
                        with open(os.path.join(self.output_dir, chunk_filename), 'w', encoding='utf-8') as f:
                            f.write(chunk_text)
                    except:
                        pass

                    try:
                        self.store_embeddings(chunk_text, chunk_topic, chunk_id)
                    except:
                        pass

                    section_data['chunks'].append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "topic": chunk_topic,
                        "entities": entities
                    })
                    doc_stats["chunks"] += 1

                document_sections.append(section_data)
            except:
                pass

            if time.time() - doc_start_time > self.doc_timeout:
                doc_stats["status"] = "partial"
                doc_stats["error"] = "Document processing timeout"
                break

        try:
            triples_created = self.create_neo4j_hierarchy(doc_name, document_sections)
            doc_stats["triples"] = triples_created
            if self.neo4j_driver:
                print(f"Loaded {triples_created} triples to Neo4j")
        except Exception as e:
            doc_stats["status"] = "partial"
            doc_stats["error"] = f"Neo4j export failed: {str(e)}"

        return doc_stats

    def save_faiss_indices(self):
        index_info = {}
        for topic, index in self.faiss_indices.items():
            index_path = os.path.join(self.faiss_index_dir, f"{topic}.index")
            meta_path = os.path.join(self.faiss_index_dir, f"{topic}_metadata.json")

            try:
                faiss.write_index(index, index_path)
                with open(meta_path, 'w') as f:
                    json.dump(self.chunk_metadata[topic], f)
                index_info[topic] = {
                    "index_path": index_path,
                    "metadata_path": meta_path,
                    "chunk_count": len(self.chunk_metadata[topic])
                }
            except:
                pass
        return index_info

    def run(self):
        results = {}
        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)

        print(f"Found {total_files} PDF files to process")
        print("-" * 60)

        total_pages = 0
        total_chunks = 0
        total_entities = 0
        total_triples = 0
        processed_files = 0

        for i, filename in enumerate(pdf_files):
            print(f"\nProcessing file {i+1}/{total_files}: {filename}")
            start_time = time.time()

            try:
                doc_result = self.process_document(filename)
                elapsed = time.time() - start_time

                if doc_result.get("status") == "skipped":
                    print(f"  SKIPPED (already processed) in {elapsed:.2f}s")
                    results[filename] = doc_result
                    continue

                total_pages += doc_result.get("pages", 0)
                total_chunks += doc_result.get("chunks", 0)
                total_entities += doc_result.get("entities", 0)
                total_triples += doc_result.get("triples", 0)
                processed_files += 1

                print(f"  Pages: {doc_result['pages']} | "
                      f"Chunks: {doc_result['chunks']} | "
                      f"Entities: {doc_result['entities']} | "
                      f"Triples: {doc_result.get('triples', 0)}")
                print(f"  Status: {doc_result['status'].upper()} in {elapsed:.2f}s")

                results[filename] = doc_result

                if (i + 1) % self.batch_size == 0:
                    print(f"\nCompleted {i+1} files, saving state...")
                    self.save_faiss_indices()
                    print(f"  Cumulative: {total_pages} pages, {total_chunks} chunks")

            except Exception as e:
                error_msg = f"Critical error: {str(e)}"
                print(error_msg)
                results[filename] = {"status": "failed", "error": error_msg}

        print("\nFinalizing processing...")
        index_info = self.save_faiss_indices()

        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY:")
        print(f"  Files processed: {processed_files}/{total_files}")
        print(f"  Total pages: {total_pages}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Total entities: {total_entities}")
        print(f"  Total triples loaded to Neo4j: {total_triples}")
        print(f"  FAISS indices created: {len(index_info)}")
        print(f"  Neo4j database: {self.neo4j_database}")

        results["summary"] = {
            "total_files": total_files,
            "processed_files": processed_files,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "total_triples": total_triples,
            "faiss_indices": index_info,
            "neo4j_database": self.neo4j_database
        }

        return results
