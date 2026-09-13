import re
import unicodedata
from difflib import SequenceMatcher
from sqlalchemy import text
from database import SessionLocal


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def verse_exists(reference: str, quoted_text: str, translation: str) -> bool:
    with SessionLocal() as db:
        row = db.execute(text("""
            SELECT text FROM verses
            WHERE reference = :ref AND translation_id = :tr LIMIT 1
        """), {"ref": reference, "tr": translation}).scalar()
    if not row:
        return False
    return SequenceMatcher(None, normalize(row), normalize(quoted_text)).ratio() > 0.85


def get_verse_by_id(verse_id: int):
    with SessionLocal() as db:
        row = db.execute(text("""
            SELECT id, reference, text, translation_id FROM verses WHERE id = :id
        """), {"id": verse_id}).mappings().first()
    return dict(row) if row else None