#!/usr/bin/env python3
import json
import pickle
from pathlib import Path
from typing import List

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

CACHE_FILE = Path("interim/ingested.pkl")
STORE_DIR = Path("vector_store")
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK = 800
OVERLAP = 100

def main() -> None:
    records: List[dict] = pickle.load(CACHE_FILE.open("rb"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK, chunk_overlap=OVERLAP)
    docs: List[Document] = []

