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
    cfg = json.loads((STORE_DIR / "params.json").read_text())
    embedder = SentenceTransformerEmbeddings(model_name=cfg["emb_model"])
    db = FAISS.load_local(str(STORE_DIR), embedder)

    llm = OpenAI(model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="map_rerank",
        retriever=db.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    )