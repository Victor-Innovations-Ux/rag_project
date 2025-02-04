# Retrieval-Augmented Generation (RAG) System

## Overview
The RAG_BOOK is a demonstration project that allows users to ask precise questions based on multiple PDF books. The system returns clear answers along with source references (including page numbers when available). It leverages LangChain for document processing, OpenAI GPT-3.5-turbo for language generation, FAISS for vector similarity search, and Streamlit for the interactive web interface.

## Features
- **PDF Ingestion**: Loads and processes PDF files.
- **Document Splitting**: Splits long documents into manageable text chunks.
- **Vector Indexing & Storage**: Uses FAISS for efficient vector-based search and **stores the index locally** to improve performance.
- **Persistent Indexing**: The vector store is cached in the `vectorstore_index/` folder, avoiding the need for reindexing on every run.
- **Question Answering**: Generates answers using OpenAI's GPT-3.5-turbo.
- **Source Attribution**: Provides source references for each generated answer.
- **Interactive Web UI**: Offers a user-friendly interface built with Streamlit.
- **Cached embeddings** : to improve startup performance and reduce redundant API calls

## Project Structure

```
RAG-System/
├── cache/                      # Folder storing cached embeddings
├── data/                       # Folder containing uploaded PDF files
├── src/                        
│   ├── preprocessing.py        # Handles PDF loading and text chunking
│   ├── vector_stores.py        # Manages FAISS vector store & cached embeddings
│   ├── query.py                # Contains the logic for generating answers
├── vectorstore_index/          # Folder storing the FAISS vector store
├── streamlit_app.py            # The main Streamlit application file
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables file (contains the OpenAI API key)
└── README.md                   # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

### 2. Install Dependencies
It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Application
To start the Streamlit application, run:
```bash
streamlit run streamlit_app.py
```
This command will launch the app in your default web browser.

### Uploading PDF Files
- Use the sidebar in the application to upload new PDF files.
- The system supports multiple PDF uploads, enabling you to ask questions based on one or several books.

### Asking Questions
- Enter your query in the text input field.
- The system processes the query and returns a clear answer along with source references (file name and page number if available).

## Persistent Vector Store
The **vector store is saved locally** in the `vectorstore_index/` directory.  
- If the book selection changes, the index is automatically **recomputed**.  
- If the index already exists, it is **loaded directly** to save computation time.  

## Example Queries
- **Single Book Query**:  
  *Question*: "At what time did the Hogwarts Express depart?"  
  *Expected Answer*: "11:00 AM (from HP1)"

- **Cross-book Query**:  
  *Question*: "What is the common factor between Harry Potter's father and Katniss's father?"  
  *Expected Answer*: "Both are deceased (information gathered from HP and HG)"

## Technologies Used
- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Python](https://www.python.org/)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
