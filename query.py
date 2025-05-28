#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

STORE_DIR = Path("vector_store")
DEFAULT_TOP_K = 6

def ask(question_words: List[str], top_k: int) -> None:
    cfg = json.loads((STORE_DIR / "params.json").read_text())
    embedder = SentenceTransformerEmbeddings(model_name=cfg["emb_model"])
    db = FAISS.load_local(
         str(STORE_DIR),
         embedder,
         allow_dangerous_deserialization=True 
     )

    llm = OpenAI(model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="map_rerank",
        retriever=db.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    )

    q = " ".join(question_words)
    result = qa(q)
    print("\n Answer:\n", result["result"], "\n")
    print("Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata)

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("question", nargs="*", help="Question text (ignored if --file is used)")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="how many chunks to retrieve")
    parser.add_argument("--file", "-f", type=str, help="Path to a .txt file with a prompt")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            ask([fh.read()], args.k)
    else:
        if not args.question:
            parser.error("Provide a question or use --file")
        ask(args.question, args.k)


if __name__ == "__main__":
    main()
    