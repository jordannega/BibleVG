"""
Run once to load verses + themes + embeddings.
Usage: python ingest.py
"""
import json
from sqlalchemy import text
from database import init_db, SessionLocal
from models import Translation, Book, Verse
from embeddings import embed
import re

THEME_SLUGS = {
    "anxiety","fear","sadness","depression","loneliness","grief","anger","guilt",
    "doubt","hope","strength","peace","love","purpose","forgiveness","gratitude",
    "temptation","illness","finance","relationship","parenting","work"
}

BOOK_LOOKUP = {
    "Genesis": ("GEN", "OT", 1), "Exodus": ("EXO", "OT", 2), "Leviticus": ("LEV", "OT", 3),
    "Numbers": ("NUM", "OT", 4), "Deuteronomy": ("DEU", "OT", 5), "Joshua": ("JOS", "OT", 6),
    "Psalms": ("PSA", "OT", 19), "Psalm": ("PSA", "OT", 19), "Proverbs": ("PRO", "OT", 20),
    "Isaiah": ("ISA", "OT", 23), "Jeremiah": ("JER", "OT", 24), "Lamentations": ("LAM", "OT", 25),
    "Micah": ("MIC", "OT", 33), "Matthew": ("MAT", "NT", 40), "Mark": ("MRK", "NT", 41),
    "John": ("JHN", "NT", 43), "Romans": ("ROM", "NT", 45), "1 Corinthians": ("1CO", "NT", 46),
    "2 Corinthians": ("2CO", "NT", 47), "Ephesians": ("EPH", "NT", 49), "Philippians": ("PHP", "NT", 50),
    "Colossians": ("COL", "NT", 51), "1 Thessalonians": ("1TH", "NT", 52), "2 Timothy": ("2TI", "NT", 55),
    "Hebrews": ("HEB", "NT", 58), "James": ("JAS", "NT", 59), "1 Peter": ("1PE", "NT", 60),
    "1 John": ("1JN", "NT", 62), "Revelation": ("REV", "NT", 66), "Numbers": ("NUM", "OT", 4),
}


def parse_reference(ref: str):
    m = re.match(r"^(.+?)\s+(\d+):(\d+)(?:-(\d+))?$", ref.strip())
    if not m:
        raise ValueError(f"Bad reference: {ref}")
    book_name, ch, vs, ve = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    ve = int(ve) if ve else vs
    return book_name, ch, vs, ve


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def main():
    init_db()

    with SessionLocal() as db:
        # Translation
        tr = db.query(Translation).get("WEB")
        if not tr:
            db.add(Translation(
                id="WEB", name="World English Bible", language="en",
                license="Public Domain", is_public_domain=True,
            ))
            db.commit()

        # Books
        for name, (code, testament, order) in BOOK_LOOKUP.items():
            if not db.query(Book).filter_by(code=code).first():
                db.add(Book(code=code, name=name, testament=testament, canonical_order=order))
        db.commit()

        # Verses
        with open("seed_verses.json") as f:
            data = json.load(f)

        new_verses = []
        for item in data:
            if db.query(Verse).filter_by(reference=item["reference"], translation_id="WEB").first():
                continue
            book_name, ch, vs, ve = parse_reference(item["reference"])
            code = BOOK_LOOKUP.get(book_name, ("GEN", "OT", 1))[0]
            book = db.query(Book).filter_by(code=code).first()
            new_verses.append(Verse(
                translation_id="WEB",
                book_id=book.id,
                chapter=ch,
                verse_start=vs,
                verse_end=ve,
                reference=item["reference"],
                text=item["text"],
                text_normalized=normalize_text(item["text"]),
                popularity=0.5,
            ))
        db.add_all(new_verses)
        db.commit()
        print(f"[ingest] Added {len(new_verses)} verses")

        # Embeddings
        rows = db.query(Verse).filter(Verse.embedding.is_(None)).all()
        if rows:
            print(f"[ingest] Embedding {len(rows)} verses...")
            BATCH = 64
            for i in range(0, len(rows), BATCH):
                chunk = rows[i:i+BATCH]
                vecs = embed([r.text for r in chunk])
                for r, v in zip(chunk, vecs):
                    r.embedding = v.tolist()
                db.commit()
                print(f"[ingest]   {i+len(chunk)}/{len(rows)}")
        print("[ingest] Done.")


if __name__ == "__main__":
    main()