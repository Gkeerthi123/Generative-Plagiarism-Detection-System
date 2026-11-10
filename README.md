
# Generative Plagiarism Detection System

An AI-powered plagiarism detection application that analyzes documents using exact word matching, phrase matching, and semantic similarity detection.

## Features

- Upload PDF, DOCX, and TXT documents.
- Add custom reference documents.
- Compare documents against PAN-PC-11 indexed source documents.
- Exact word matching.
- Repeated phrase detection.
- Semantic similarity detection using Sentence Transformers and FAISS.
- AI-powered analysis summary using Ollama (Llama 3.2).
- Similarity dashboard and charts using Plotly.
- Side-by-side matching evidence.
- Downloadable PDF plagiarism reports.

## Technologies Used

- Python 3.12
- Streamlit
- Sentence Transformers
- FAISS
- Ollama – Llama 3.2
- SQLite / Python utilities
- Plotly
- PyPDF / Document processing libraries
- ReportLab

## Project Structure

```text
Generative_Plagiarism_Detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── pan_pc11/
│   ├── processed/
│   └── custom_references/
│
├── models/
│   ├── source_index.faiss
│   └── source_chunks.pkl
│
├── src/
│   ├── data_loader.py
│   ├── text_preprocessing.py
│   ├── embedding.py
│   ├── semantic_search.py
│   ├── plagiarism_detector.py
│   ├── document_reader.py
│   ├── reference_manager.py
│   ├── text_matching.py
│   ├── pdf_report.py
│   └── ai_summary.py
│
├── tests/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Generative_Plagiarism_Detectio
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install and start Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running.

### 6. Run the application

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal.

## How to Use

1. Open the application.
2. Upload original reference documents.
3. Upload the document you want to check.
4. Click **Check for Plagiarism**.
5. Review similarity scores and matching passages.
6. Read the AI-powered analysis summary.
7. Download the PDF report.

## How It Works

1. Document Extraction – Extracts text from uploaded documents.
2. Text Preprocessing – Cleans and prepares extracted text.
3. Exact Matching – Identifies common words and phrases.
4. Semantic Analysis – Generates embeddings and compares document meaning.
5. Similarity Detection – Identifies potentially similar content.
6. AI Analysis – Generates a readable analysis summary.
7. PDF Reporting – Creates a downloadable analysis report.

## Important Note

Similarity scores are indicators and do not automatically prove plagiarism. Results should be reviewed manually.

The application compares documents against the indexed PAN-PC-11 sources and user-uploaded reference documents. It does not search the entire internet.

## Author

Developed as an AI and Machine Learning project.
