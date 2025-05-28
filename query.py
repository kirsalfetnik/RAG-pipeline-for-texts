#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List
from fpdf import FPDF
from datetime import datetime
import re

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA

STORE_DIR = Path("vector_store")
DEFAULT_TOP_K = 6

def sanitize(text: str) -> str:
    text = (
        text
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("…", "...")
        .replace("–", "-")    
        .replace("—", "--")   
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("↓", "v")
        .replace("↑", "^")
    )
    return re.sub(r"[^\x20-\x7E\r\n]", "", text)

def ask(question_words: List[str], top_k: int) -> None:
    cfg = json.loads((STORE_DIR / "params.json").read_text())
    embedder = SentenceTransformerEmbeddings(model_name=cfg["emb_model"])
    db = FAISS.load_local(
         str(STORE_DIR),
         embedder,
         allow_dangerous_deserialization=True 
     )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large") # Alternative model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large") # Alternative model
    hf_pipe = pipeline( # Alternative model
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_length=512,
        truncation=True  
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe) # Alternative model
    # llm = ChatOpenAI(model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff", # "map_rerank" by default
        retriever=db.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    )

    q = " ".join(question_words)
    result = qa(q)
    print("\n Answer:\n", result["result"], "\n")
    print("Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata)

    # Save PDF-file
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "RAG Pipeline Answer", ln=True, align="C")
    pdf.ln(5)
    # Input
    question_text = " ".join(question_words)
    question_text = sanitize(" ".join(question_words))
    pdf.multi_cell(pdf.epw, 8, f"Question: {question_text}")
    pdf.ln(2)
    # Output
    answer_text = sanitize(result["result"])
    pdf.multi_cell(pdf.epw, 8, f"Answer:\n{answer_text}")
    pdf.ln(4)
    # Sources
    pdf.multi_cell(pdf.epw, 8, "Sources:")
    for doc in result["source_documents"]:
        meta = "; ".join(f"{k}={v}" for k, v in doc.metadata.items())
        safe_line = sanitize(f"- {meta}")
        pdf.multi_cell(pdf.epw, 6, safe_line)

    # Timestamps
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = STORE_DIR.parent / f"answer_{ts}.pdf"
    pdf.output(str(out_path))
    print(f"\n Answer saved to {out_path}")

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
    