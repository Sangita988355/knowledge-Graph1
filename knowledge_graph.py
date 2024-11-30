import spacy
import networkx as nx
from pyvis.network import Network
import random
import pdfplumber
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Function to extract entities using spaCy's NER
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return list(set(entities))  # Removing duplicates

# Machine Learning-based Relation Extraction (using Random Forest for simplicity)
# For the sake of this example, we use a simple Random Forest classifier. 
# In practice, you'd train this on labeled data for better performance.
def extract_relationships_with_ml(text, entities):
    relationships = []
    
    # Prepare training data (for example, just a few training pairs for demonstration)
    # In practice, this should be trained with a larger labeled dataset of sentence pairs and relationships.
    data = [
        ("John works at Google", "works_at"),
        ("Mary works at Microsoft", "works_at"),
        ("Sarah is located in California", "located_in"),
        ("Amazon is in Seattle", "located_in"),
        ("Steve is a CEO of Apple", "has_role"),
    ]
    
    # Create feature extraction for sentences (using TF-IDF)
    sentences = [pair[0] for pair in data]
    labels = [pair[1] for pair in data]
    
    # Train a simple classifier
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    y = labels
    
    # Split data and train the classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict relationship between entities in the given text
    for entity in entities:
        for sentence in text.split('.'):
            if entity in sentence:
                # Extract features from the sentence
                sentence_vec = vectorizer.transform([sentence])
                relationship_pred = clf.predict(sentence_vec)
                relationships.append((entity, relationship_pred[0], sentence))
    
    return relationships

# Function to create the knowledge graph using NetworkX
def create_knowledge_graph(entities, relationships):
    G = nx.Graph()
    
    # Add nodes with entity types
    for entity in entities:
        G.add_node(entity, type="Entity")
    
    # Add relationships as edges with labels
    for subj, rel, obj in relationships:
        G.add_edge(subj, obj, label=rel)
    
    return G

# Function to visualize the knowledge graph with multi-colored nodes
def visualize_graph(graph):
    net = Network(height="600px", width="100%", notebook=True)
    
    # Color palette for nodes
    colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF1493', '#00FFFF', '#FF4500', '#228B22', '#D2691E']
    
    # Add nodes with random colors
    for node, data in graph.nodes(data=True):
        color = random.choice(colors)
        net.add_node(node, label=node, title=data.get("type", "Entity"), color=color)
    
    # Add edges with labels
    for u, v, data in graph.edges(data=True):
        net.add_edge(u, v, label=data['label'])
    
    # Customize layout and appearance
    net.force_atlas_2based()
    net.show("graph.html")

# Streamlit UI
st.title("Machine Learning Enhanced Knowledge Graph Generator")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner('Processing...'):
        try:
            file_extension = uploaded_file.name.split(".")[-1]
            
            if file_extension == "pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")
            
            # Show extracted text
            st.write("Extracted Text:")
            st.write(text)
            
            # Extract entities using spaCy
            entities = extract_entities(text)
            
            if not entities:
                st.error("No entities were extracted.")
            
            st.write("Entities:", entities)
            
            # Extract relationships using the trained ML model
            relationships = extract_relationships_with_ml(text, entities)
            
            if not relationships:
                st.error("No relationships were extracted.")
            
            st.write("Relationships:", relationships)
            
            # Create the knowledge graph
            graph = create_knowledge_graph(entities, relationships)
            
            # Visualize the graph
            visualize_graph(graph)
            
            # Show graph in Streamlit
            st.components.v1.html(open("graph.html", "r").read(), height=800)
        
        except Exception as e:
            st.error(f"Error: {e}")
