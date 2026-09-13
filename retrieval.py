from sqlalchemy import text
from database import SessionLocal


VECTOR_SQL = text("""
    SELECT v.id, v.reference, v.text, v.translation_id,
           1 - (v.embedding <=> CAST(:qvec AS vector)) AS similarity
    FROM verses v
    WHERE v.translation_id = :translation
    ORDER BY v.embedding <=> CAST(:qvec AS vector)
    LIMIT :k
""")

LEXICAL_SQL = text("""
    SELECT v.id, v.reference, v.text, v.translation_id,
           ts_rank_cd(to_tsvector('english', v.text), plainto_tsquery('english', :q)) AS score
    FROM verses v
    WHERE v.translation_id = :translation
      AND to_tsvector('english', v.text) @@ plainto_tsquery('english', :q)
    ORDER BY score DESC
    LIMIT :k
""")


def vector_search(qvec, translation="WEB", k=30):
    with SessionLocal() as db:
        rows = db.execute(VECTOR_SQL, {"qvec": str(qvec), "translation": translation, "k": k}).mappings().all()
    return [dict(r) for r in rows]


def lexical_search(q, translation="WEB", k=30):
    with SessionLocal() as db:
        rows = db.execute(LEXICAL_SQL, {"q": q, "translation": translation, "k": k}).mappings().all()
    return [dict(r) for r in rows]


def rrf(rank_lists, k=60, top_n=10):
    scores = {}
    lookup = {}
    for ranks in rank_lists:
        for i, item in enumerate(ranks):
            vid = item["id"]
            scores[vid] = scores.get(vid, 0) + 1.0 / (k + i + 1)
            lookup[vid] = item
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [{**lookup[vid], "rrf_score": s} for vid, s in ranked]