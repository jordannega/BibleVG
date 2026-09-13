from enum import Enum
from pydantic import BaseModel
from typing import Optional
import json


class Theme(str, Enum):
    ANXIETY = "anxiety"
    FEAR = "fear"
    SADNESS = "sadness"
    DEPRESSION = "depression"
    LONELINESS = "loneliness"
    GRIEF = "grief"
    ANGER = "anger"
    GUILT = "guilt"
    DOUBT = "doubt"
    HOPE = "hope"
    STRENGTH = "strength"
    PEACE = "peace"
    LOVE = "love"
    PURPOSE = "purpose"
    FORGIVENESS = "forgiveness"
    GRATITUDE = "gratitude"
    TEMPTATION = "temptation"
    ILLNESS = "illness"
    FINANCE = "finance"
    RELATIONSHIP = "relationship"
    PARENTING = "parenting"
    WORK = "work"
    GENERAL = "general"


class Intent(BaseModel):
    primary_theme: Theme
    subtype: Optional[str] = None
    is_question: bool = False
    wants_prayer: bool = False
    confidence: float = 0.0


KEYWORDS = {
    Theme.ANXIETY: ["anxious", "anxiety", "worry", "worried", "panic", "overwhelmed", "stress", "stressed"],
    Theme.FEAR: ["afraid", "scared", "fear", "terrified", "phobia"],
    Theme.SADNESS: ["sad", "unhappy", "blue", "heartbroken", "crying", "down"],
    Theme.DEPRESSION: ["depress", "empty", "numb", "hopeless", "cant get out of bed"],
    Theme.LONELINESS: ["lonely", "alone", "isolated", "abandoned"],
    Theme.GRIEF: ["died", "passed away", "funeral", "loss", "grieving", "mourning", "death"],
    Theme.ANGER: ["angry", "anger", "furious", "rage", "mad"],
    Theme.GUILT: ["guilt", "guilty", "shame", "ashamed", "sinned", "sinning"],
    Theme.DOUBT: ["doubt", "unbelief", "is god real", "why god", "questioning"],
    Theme.HOPE: ["hope", "future", "discouraged", "dream", "despair"],
    Theme.STRENGTH: ["tired", "weak", "exhausted", "burnout", "cant go on"],
    Theme.PEACE: ["peace", "calm", "rest", "still"],
    Theme.LOVE: ["love", "loved", "marriage", "relationship", "friend"],
    Theme.PURPOSE: ["purpose", "meaning", "direction", "calling", "what should i do"],
    Theme.FORGIVENESS: ["forgive", "forgiveness"],
    Theme.GRATITUDE: ["thankful", "grateful", "gratitude", "blessed"],
    Theme.TEMPTATION: ["temptation", "tempted", "addiction", "struggle"],
    Theme.ILLNESS: ["sick", "diagnosis", "cancer", "hospital", "pain", "healing"],
    Theme.FINANCE: ["money", "debt", "bills", "unemployed", "rent", "poor"],
    Theme.RELATIONSHIP: ["husband", "wife", "boyfriend", "girlfriend", "divorce"],
    Theme.PARENTING: ["child", "kid", "son", "daughter", "parenting"],
    Theme.WORK: ["boss", "work", "career", "interview", "coworker", "job"],
}


def classify_keyword(text: str) -> Intent:
    lowered = text.lower()
    scores = {}
    for theme, kws in KEYWORDS.items():
        s = sum(1 for kw in kws if kw in lowered)
        if s:
            scores[theme] = s
    if not scores:
        return Intent(primary_theme=Theme.GENERAL, confidence=0.3)
    best = max(scores, key=scores.get)
    return Intent(primary_theme=best, confidence=min(0.9, 0.3 + scores[best] * 0.15))