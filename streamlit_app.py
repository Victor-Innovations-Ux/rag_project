##
## FREE PROJECT, 2025
## RAG_SYSTEM DEMO
## File description:
## Streamlit app
##

import os
import streamlit as st
from dotenv import load_dotenv

from src.preprocessing import load_pdf, split_documents
from src.vector_stores import create_or_load_vectorstore
from src.query import answer_query

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY isn't defined.")
    st.stop()

st.title("Système RAG pour les Livres")
st.write("Posez une question sur le contenu des livres et obtenez une réponse avec les sources.")

if "available_books" not in st.session_state:
    st.session_state.available_books = [
        "data/HP1.pdf",
        "data/HP2.pdf",
        "data/HG.pdf"
    ]

st.sidebar.subheader("Ajouter un nouveau fichier")
uploaded_file = st.sidebar.file_uploader("Télécharger un fichier (PDF)", type=["pdf"])
if uploaded_file is not None:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Fichier {uploaded_file.name} uploadé avec succès !")
        if file_path not in st.session_state.available_books:
            st.session_state.available_books.append(file_path)
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement du fichier : {e}")

selected_books = st.sidebar.multiselect(
    "Choisissez les livres",
    st.session_state.available_books,
    default=st.session_state.available_books
)

if "vectorstore" not in st.session_state or set(selected_books) != set(st.session_state.get("selected_books", [])):
    st.info("Chargement et indexation des documents en cours...")
    all_chunks = []
    for pdf_path in selected_books:
        docs = load_pdf(pdf_path)
        for doc in docs:
            doc.metadata["source"] = pdf_path
        chunks = split_documents(docs)
        all_chunks.extend(chunks)
    
    if all_chunks:
        vectorstore = create_or_load_vectorstore(
            all_chunks,
            openai_api_key,
            index_path="vectorstore_index",
            force_recreate=False
        )
        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            st.session_state.selected_books = selected_books
            st.success("Index chargé avec succès !")
        else:
            st.error("Erreur lors de la création de l'index vectoriel.")
    else:
        st.error("Aucun chunk généré à partir des documents.")

query = st.text_input("Votre question :", "")
if st.button("Envoyer") and query:
    with st.spinner("Recherche de la réponse..."):
        result = answer_query(query, st.session_state.vectorstore, openai_api_key)
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])
        top_docs = source_docs[:3]
        sources_info = []
        for doc in top_docs:
            source = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", "N/A")
            source_name = os.path.basename(source)
            sources_info.append(f"{source_name} - page {page}")
        st.subheader("Réponse générée :")
        st.write(f"{answer}  _(Sources : {', '.join(sources_info)})_")
