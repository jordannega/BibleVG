from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, Float, DateTime, BigInteger
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import uuid
from database import Base
from config import EMBED_DIM


class Translation(Base):
    __tablename__ = "translations"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    language = Column(String, nullable=False)
    license = Column(String, nullable=False)
    is_public_domain = Column(Boolean, default=False)


class Book(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    testament = Column(String)
    canonical_order = Column(Integer)


class Verse(Base):
    __tablename__ = "verses"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    translation_id = Column(String, ForeignKey("translations.id"), index=True)
    book_id = Column(Integer, ForeignKey("books.id"), index=True)
    chapter = Column(Integer, nullable=False)
    verse_start = Column(Integer, nullable=False)
    verse_end = Column(Integer, nullable=False)
    reference = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    text_normalized = Column(Text, nullable=False)
    embedding = Column(Vector(EMBED_DIM))
    popularity = Column(Float, default=0.0)


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_hash = Column(String, index=True)
    prompt_redacted = Column(Text)
    detected_theme = Column(String)
    crisis_flag = Column(Boolean, default=False)
    verse_id = Column(BigInteger, ForeignKey("verses.id"))
    model_used = Column(String)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Integer)
    feedback_score = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())