# rag_pipeline/ingest.py

import re, sys, json, uuid, camelot, pdfplumber, tabula
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from slugify import slugify
import pickle

DATA_DIR = Path("data")    # pdf files are here
OUT_FILE = Path("interim/ingested.pkl")  

OUT_FILE.parent.mkdir(exist_ok=True, parents=True)