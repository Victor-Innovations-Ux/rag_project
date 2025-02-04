##
## FREE PROJECT, 2025
## RAG_SYSTEM DEMO
## File description:
## Create vector store
##

import os
import pickle
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import PrivateAttr

def get_cache_path(filename: str = "embeddings_cache.pkl") -> str:
    cache_dir = os.path.join(os.getcwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)

class CachedOpenAIEmbeddings(OpenAIEmbeddings):
    _cache_file: str = PrivateAttr()
    _cache: dict = PrivateAttr()

    def __init__(self, cache_file=None, *args, **kwargs):
        if cache_file is None:
            cache_file = get_cache_path()
        super().__init__(*args, **kwargs)
        self._cache_file = cache_file
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception as e:
                self._cache = {}
        else:
            self._cache = {}

    def _hash_text(self, text: str) -> str:
        normalized_text = " ".join(text.strip().split())
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts):
        results = []
        missing_texts = []
        missing_indices = []
        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)
                missing_texts.append(text)
                missing_indices.append(i)
        if missing_texts:
            computed_embeddings = super().embed_documents(missing_texts)
            for idx, embedding in zip(missing_indices, computed_embeddings):
                results[idx] = embedding
                key = self._hash_text(texts[idx])
                self._cache[key] = embedding
            try:
                with open(self._cache_file, "wb") as f:
                    pickle.dump(self._cache, f)
            except Exception as e:
                print(f"[CACHE] Erreur lors de la sauvegarde du cache dans {self._cache_file} : {e}")
        return results

    def embed_query(self, text: str):
        key = self._hash_text(text)
        if key in self._cache:
            return self._cache[key]
        embedding = super().embed_query(text)
        self._cache[key] = embedding
        try:
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"[CACHE] Erreur lors de la sauvegarde du cache après traitement de la requête : {e}")
        return embedding

def create_or_load_vectorstore(chunks, openai_api_key, index_path="vectorstore_index", force_recreate=False):
    if not force_recreate and os.path.exists(index_path):
        try:
            embeddings = CachedOpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=openai_api_key
            )
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            print(f"[INDEX] Erreur lors du chargement de l'index vectoriel : {e}")
    try:
        embeddings = CachedOpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"[INDEX] Erreur lors de la création du vector store : {e}")
        raise
