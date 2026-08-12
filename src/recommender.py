"""
recommender.py
---------------
Core content-based recommendation logic. No external dependencies and no
API key required -- this module works entirely offline using simple,
explainable math (bag-of-words vectors + cosine similarity).

How it works, in plain English:
1. Every product is turned into a "feature vector": a word-count dictionary
   built from its category, tags, and description (category and tags are
   weighted higher because they're stronger signals than free-text
   description).
2. Every user is turned into a feature vector too, built from their stated
   preferences (categories + tags). If the user has a purchase history,
   we blend the average vector of their purchased products with their
   stated preferences, so recommendations improve as we learn about them.
3. We rank every unpurchased, filter-passing product by cosine similarity
   to the user's vector, add small rule-based bonuses (exact category
   match, in-budget), and return the top N with a plain-English reason.

This is a "content-based filtering" approach: it compares product content
to user-stated interests, rather than needing other users' behavior
(collaborative filtering), which makes it robust to cold-start users.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# A tiny stopword list -- just enough to keep description tokens meaningful.
STOPWORDS = {
    "a", "an", "the", "for", "and", "with", "of", "to", "your", "that",
    "is", "in", "on", "from", "this", "it", "are", "you", "your", "into",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into words, drop stopwords/short words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


@dataclass
class Product:
    id: str
    name: str
    category: str
    brand: str
    price: float
    tags: List[str]
    description: str

    def feature_vector(self) -> Counter:
        """Build a weighted bag-of-words vector for this product."""
        tokens: List[str] = []
        tokens += [self.category.lower()] * 3          # category: strong signal
        for tag in self.tags:
            tokens += [tag.lower()] * 2                 # tags: strong signal
        tokens += tokenize(self.description)             # description: light signal
        return Counter(tokens)


@dataclass
class UserProfile:
    id: str
    name: str
    preferred_categories: List[str] = field(default_factory=list)
    preferred_tags: List[str] = field(default_factory=list)
    disliked_tags: List[str] = field(default_factory=list)
    budget_max: Optional[float] = None
    budget_min: Optional[float] = None
    purchase_history: List[str] = field(default_factory=list)

    def stated_preference_vector(self) -> Counter:
        """Build a vector purely from what the user says they like."""
        tokens: List[str] = []
        for c in self.preferred_categories:
            tokens += [c.lower()] * 3
        for t in self.preferred_tags:
            tokens += [t.lower()] * 2
        return Counter(tokens)


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Standard cosine similarity between two sparse bag-of-words vectors."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Recommender:
    def __init__(self, products: List[Product]):
        self.products: Dict[str, Product] = {p.id: p for p in products}
        self._vectors: Dict[str, Counter] = {p.id: p.feature_vector() for p in products}

    def build_user_vector(self, user: UserProfile, history_weight: float = 0.6) -> Counter:
        """
        Cold start (no purchase history): use stated preferences only.
        Returning user: blend purchase-history vector with stated preferences,
        weighted by `history_weight` (behavior counts more than words, but
        stated preferences still matter -- e.g. new interests the user typed
        in but hasn't bought yet).
        """
        stated = user.stated_preference_vector()
        if not user.purchase_history:
            return stated

        history_vec: Counter = Counter()
        for pid in user.purchase_history:
            if pid in self._vectors:
                history_vec.update(self._vectors[pid])

        blended: Counter = Counter()
        for k, v in history_vec.items():
            blended[k] += v * history_weight
        for k, v in stated.items():
            blended[k] += v * (1 - history_weight)
        return blended

    def recommend(self, user: UserProfile, top_n: int = 5) -> List[dict]:
        user_vec = self.build_user_vector(user)
        cold_start = len(user.purchase_history) == 0
        disliked = {t.lower() for t in user.disliked_tags}
        preferred_cats = {c.lower() for c in user.preferred_categories}

        scored = []
        for pid, product in self.products.items():
            if pid in user.purchase_history:
                continue  # don't recommend what they already own

            product_tags = {t.lower() for t in product.tags}
            if disliked and (product_tags & disliked):
                continue  # hard filter: never show disliked attributes

            if user.budget_max is not None and product.price > user.budget_max * 1.15:
                continue  # allow a little flexibility, then hard-filter

            score = cosine_similarity(user_vec, self._vectors[pid])

            if product.category.lower() in preferred_cats:
                score += 0.05  # small bonus for an exact category match
            if user.budget_max is not None and product.price <= user.budget_max:
                score += 0.03  # small bonus for comfortably fitting budget

            scored.append((product, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_n]

        return [
            {
                "product": product,
                "score": round(score, 4),
                "reason": self._explain(user, product, score, cold_start),
            }
            for product, score in top
        ]

    def _explain(self, user: UserProfile, product: Product, score: float, cold_start: bool) -> str:
        """Rule-based, fully offline explanation of why a product was recommended."""
        product_tags = {t.lower() for t in product.tags}
        preferred_tags = {t.lower() for t in user.preferred_tags}
        matched_tags = sorted(product_tags & preferred_tags)
        matched_category = product.category.lower() in {c.lower() for c in user.preferred_categories}

        reasons = []
        if matched_category:
            reasons.append(f"matches your preferred category '{product.category}'")
        if matched_tags:
            reasons.append(f"shares your interests: {', '.join(matched_tags)}")
        if user.budget_max is not None:
            if product.price <= user.budget_max:
                reasons.append(f"fits within your ${user.budget_max:.0f} budget")
            else:
                reasons.append("slightly above budget but closely matches your taste")

        if not reasons:
            if cold_start:
                reasons.append("a well-rounded pick in a category you're exploring (cold start default)")
            else:
                reasons.append("similar overall profile to items you've bought before")

        return f"Recommended ({score * 100:.1f}% match): " + "; ".join(reasons) + "."
