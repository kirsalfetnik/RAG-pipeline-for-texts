#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

STORE_DIR = Path("vector_store")
DEFAULT_TOP_K = 6

def ask(question_words: List[str], top_k: int) -> None:
    pass