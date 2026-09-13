import hashlib
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import init_db, SessionLocal
from safety import detect_crisis, CRISIS_TEMPLATE, scrub_pii, should_refuse
from classifier import classify_keyword
from embeddings import embed_one
from retrieval import vector_search, lexical_search, rrf
from verifier import verse_exists, get_verse_by_id
from llm import select_verse, reflect
from models import Interaction
from config import DEFAULT_TRANSLATION

app = FastAPI(title="Bible Verse AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def startup():
    init_db()


class VerseRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    translation: str = DEFAULT_TRANSLATION


class VerseResponse(BaseModel):
    kind: str
    theme: str | None = None
    reference: str | None = None
    text: str | None = None
    translation: str | None = None
    reflection: str | None = None
    message: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/verse", response_model=VerseResponse)
def generate_verse(req: VerseRequest):
    t0 = time.time()

    # 1. Crisis override — always first
    signal = detect_crisis(req.prompt)
    if signal.triggered and signal.severity == "critical":
        _log_interaction(req.prompt, "crisis", None, True, t0, 0.0)
        return VerseResponse(kind="crisis", message=CRISIS_TEMPLATE)

    # 2. Refuse harmful misuse
    if should_refuse(req.prompt):
        _log_interaction(req.prompt, "refused", None, False, t0, 0.0)
        return VerseResponse(
            kind="refused",
            message="I can't help find verses to justify harm. If you're hurting, please reach out to someone you trust.",
        )

    # 3. Classify
    intent = classify_keyword(req.prompt)

    # 4. Hybrid retrieval
    qvec = embed_one(req.prompt)
    v = vector_search(qvec, req.translation, k=30)
    l = lexical_search(req.prompt, req.translation, k=30)
    fused = rrf([v, l], top_n=8)

    if not fused:
        raise HTTPException(status_code=404, detail="No verses found")

    # 5. LLM selection
    try:
        sel = select_verse(req.prompt, intent.primary_theme.value, fused)
        chosen_id = sel.get("verse_id", fused[0]["id"])
    except Exception as e:
        print(f"[llm select failed] {e}")
        chosen_id = fused[0]["id"]

    chosen = next((c for c in fused if c["id"] == chosen_id), fused[0])

    # 6. Verify chosen verse exists in DB (hallucination guard)
    if not verse_exists(chosen["reference"], chosen["text"], req.translation):
        # fall back to first verified
        chosen = next((c for c in fused if verse_exists(c["reference"], c["text"], req.translation)), fused[0])

    # 7. Reflect
    try:
        reflection = reflect(req.prompt, intent.primary_theme.value, chosen, req.translation)
    except Exception as e:
        print(f"[llm reflect failed] {e}")
        reflection = f"May this verse meet you where you are today."

    # 8. Log
    latency_ms = int((time.time() - t0) * 1000)
    _log_interaction(req.prompt, intent.primary_theme.value, chosen["id"], False, t0, 0.0)

    return VerseResponse(
        kind="verse",
        theme=intent.primary_theme.value,
        reference=chosen["reference"],
        text=chosen["text"],
        translation=req.translation,
        reflection=reflection,
    )


def _log_interaction(prompt, theme, verse_id, crisis, t0, cost):
    try:
        with SessionLocal() as db:
            db.add(Interaction(
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt_redacted=scrub_pii(prompt),
                detected_theme=theme,
                crisis_flag=crisis,
                verse_id=verse_id,
                model_used="claude-sonnet-4-5",
                cost_usd=cost,
                latency_ms=int((time.time() - t0) * 1000),
            ))
            db.commit()
    except Exception as e:
        print(f"[log failed] {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)