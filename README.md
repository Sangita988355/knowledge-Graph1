# 🧠 Machine Learning Knowledge Graph Generator

This is a simple Streamlit web app that creates a **Knowledge Graph** from any **PDF or text file** using **Machine Learning and NLP (Natural Language Processing)**.  
It extracts important entities (like names, organizations, and places) and finds possible relationships between them — then visualizes everything as an interactive graph.

---

## 🚀 What It Can Do
- 📄 Reads text or PDFs  
- 🧠 Finds names, companies, and places using **spaCy**  
- 🤖 Uses a small **Random Forest model** to guess relationships between entities  
- 🕸️ Builds a **Knowledge Graph** automatically  
- 🎨 Shows an interactive graph right inside Streamlit  

---

## ⚙️ Tools Used

| Purpose | Library |
|----------|----------|
| NLP | spaCy |
| Machine Learning | Scikit-learn (RandomForest) |
| Graph Visualization | NetworkX + PyVis |
| Web App | Streamlit |
| PDF Reader | pdfplumber |

---

## 💡 How It Works
1. You upload a **PDF or text file**.  
2. The app extracts all text using `pdfplumber`.  
3. `spaCy` scans the text and finds named entities.  
4. A lightweight **Random Forest classifier** predicts how these entities might be related.  
5. The app builds a **Knowledge Graph** with these entities and relationships.  
6. You can explore the graph interactively in your browser.

---

## 🧩 Project Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/knowledge-graph-generator.git
cd knowledge-graph-generator

Install Dependencies

Create a virtual environment (optional) and install the requirements:

pip install -r requirements.txt

3️⃣ Download the spaCy Model
python -m spacy download en_core_web_sm

▶️ Run the App

Start the Streamlit app:

streamlit run app.py


Then open your browser at 👉 http://localhost:8501

📊 Example Output

Extracted Entities:

['Google', 'John', 'Seattle', 'Amazon']


Predicted Relationships:

[('John', 'works_at', 'Google'), ('Amazon', 'located_in', 'Seattle')]


Result:
An interactive knowledge graph showing connections between people, organizations, and places.

📁 Folder Structure
📂 knowledge-graph-generator
 ┣ 📜 app.py              # Main Streamlit app
 ┣ 📜 README.md           # This file
 ┣ 📜 requirements.txt    # Dependencies
 ┗ 📂 sample_data/        # Optional demo files

🌱 Future Plans

Use a larger ML model (like BERT) for better relationship accuracy

Color-code nodes by entity type (Person, Org, Location)

Export graph as PNG, JSON, or GraphML

Add support for multi-page and scanned PDFs

🤝 Contributing

If you’d like to improve this project:

Fork the repository

Make your changes

Create a pull request 🚀

All suggestions and ideas are welcome!

🧾 License

This project is under the MIT License — feel free to use and modify it.

👩‍💻 Author
Sangita Gorai
💌 sangitagorai988@gmail.com

🌐Sangita988355
