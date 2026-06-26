import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from safetensors.torch import load_model
from tokenizers import Tokenizer

from zyluncpt_model import ZylunCPT, cfg


MODEL_DIR = Path(os.getenv("MODEL_DIR", "./model"))
MODEL_FILE = os.getenv("MODEL_FILE", "model.safetensors")
TOKENIZER_FILE = os.getenv("TOKENIZER_FILE", "zyluncpt_tokenizer.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.35"))
TOP_K = int(os.getenv("TOP_K", "30"))


def require_file(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(
            f"{label} não encontrado em: {path}. "
            f"Coloque o arquivo dentro da pasta model/ ou ajuste MODEL_DIR."
        )


tokenizer_path = MODEL_DIR / TOKENIZER_FILE
model_path = MODEL_DIR / MODEL_FILE

require_file(tokenizer_path, "Tokenizer")
require_file(model_path, "Modelo")

print(f"[ZylunCPT] Carregando tokenizer: {tokenizer_path}")
tokenizer = Tokenizer.from_file(str(tokenizer_path))

print(f"[ZylunCPT] Carregando modelo: {model_path}")
print(f"[ZylunCPT] DEVICE={DEVICE}")

model = ZylunCPT(cfg).to(DEVICE)
load_model(model, str(model_path))
model.eval()


app = FastAPI(title="ZylunCPT 0.1 API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatHistoryItem(BaseModel):
    user: str = ""
    assistant: str = ""


class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem] = []


def build_prompt(message: str, history: List[ChatHistoryItem]) -> str:
    prompt = """<bos><system>
Você é ZylunCPT 0.1, uma IA brasileira de chat.
Responda sempre em português do Brasil, exceto se o usuário pedir outro idioma.
Seja direto, útil e natural.
</system>
"""

    for item in history[-4:]:
        if item.user and item.assistant:
            prompt += f"<user>\n{item.user}\n</user>\n"
            prompt += f"<assistant>\n{item.assistant}\n</assistant>\n"

    prompt += f"<user>\n{message}\n</user>\n<assistant>\n"
    return prompt


def clean_output(text: str) -> str:
    for stop in ["</assistant>", "<user>", "<system>", "<eos>"]:
        text = text.split(stop)[0]

    text = text.replace("<assistant>", "")
    return text.strip()


@torch.no_grad()
def generate_answer(message: str, history: List[ChatHistoryItem]) -> str:
    prompt = build_prompt(message, history)

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    max_context = getattr(model.cfg, "seq_len", 1024)
    input_ids = input_ids[:, -max_context:]

    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    new_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode([int(i) for i in new_ids])
    text = clean_output(text)

    if not text:
        text = "Ainda estou aprendendo. Tente perguntar de outro jeito."

    return text


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "status": "online",
        "model": "ZylunCPT 0.1",
        "device": DEVICE,
        "endpoint": "/chat"
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, str]:
    msg = req.message.strip()

    if not msg:
        raise HTTPException(status_code=400, detail="message vazio")

    answer = generate_answer(msg, req.history)
    return {"answer": answer}
