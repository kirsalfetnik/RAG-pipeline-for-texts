# list_rub_models.py
import os, requests, json, textwrap
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())          
token = os.getenv("RUBGPT_TOKEN")
if not token:
    raise SystemExit("RUBGPT_TOKEN is not set")

url = "https://gpt.ruhr-uni-bochum.de/external/v1/models"
headers = {"Authorization": f"Bearer {token}"}

resp = requests.get(url, headers=headers, timeout=30)
resp.raise_for_status()             # упадём, если ответ не 2xx
models = resp.json()

print(f"\n{len(models)} models available:\n")
for m in models:
    name   = m.get("name",   "-")
    prov   = m.get("provider", m.get("source", "-"))
    ctx    = m.get("context_length", m.get("max_context", "-"))
    in_tok = m.get("price_in",  "-")
    out_tok= m.get("price_out", "-")

    line = textwrap.shorten(
        f"{name:30} prov={prov:10} ctx={ctx:>6}  in={in_tok}  out={out_tok}",
        width=120, placeholder=" …"
    )
    print(line)

# Если хотите посмотреть полный JSON:
# print(json.dumps(models, indent=2, ensure_ascii=False))
