# Product Recommendation System

**One-sentence job statement:**
> My agent takes a user's stated preferences (and optional purchase history) and produces a ranked list of product recommendations, each with a plain-English reason.

This is a **content-based filtering** agent: it compares a product catalogue against a user's stated interests (and past purchases, if any) using cosine similarity over weighted bag-of-words vectors. The core logic needs **no API key and no internet connection** — an LLM (Claude) is used only as an *optional* layer to rephrase the reasons in friendlier language.

---

## Project Structure

```
product-recommendation-agent/
├── README.md                 # this file
├── requirements.txt          # optional dependencies (anthropic, python-dotenv)
├── .env.example               # copy to .env to enable the optional LLM layer
├── data/
│   ├── products.json          # 22-item product catalogue
│   └── users.json              # 4 sample user profiles
├── src/
│   ├── recommender.py          # core content-based filtering logic (no dependencies)
│   ├── llm_explainer.py        # optional Claude-powered reason rewriting
│   └── main.py                 # CLI entry point / demo runner
├── tests/
│   └── test_recommender.py     # assert-based tests, no pytest required
└── outputs/
    └── sample_output.txt       # generated automatically each time you run main.py
```

---

## 1. Install

Requires Python 3.9+.

```bash
cd product-recommendation-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt  # optional, only needed for the LLM layer
```

If you skip `pip install`, the agent still works — it just won't have the `anthropic` or `python-dotenv` packages, and will silently use rule-based reasons.

## 2. Configure (optional)

Only needed if you want Claude to rephrase recommendation reasons in a friendlier voice.

```bash
cp .env.example .env
# then edit .env and paste in your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at https://console.anthropic.com/. No key? No problem — leave `.env` alone and the agent runs entirely offline with rule-based reasons.

## 3. Run

```bash
python3 src/main.py
```

This will:
1. Load the product catalogue and all 4 sample user profiles.
2. Print top-5 recommendations (with reasons) for each user.
3. Save the full output to `outputs/sample_output.txt`.
4. Ask if you'd like to enter your own preferences interactively (type `y` to try it, `n` to exit).

## 4. Run the tests

```bash
python3 tests/test_recommender.py
```

All four tests should print `PASS` — they check cold-start ranking, disliked-tag filtering, purchase-history exclusion, and budget filtering.

---

## Design & Approach

### Similarity method
Each product and user is converted into a **weighted bag-of-words vector** (a word-count dictionary), then compared using **cosine similarity**:

- Product vectors are built from: category (×3 weight), tags (×2 weight each), and tokenized description (×1 weight). Category and tags are stronger, more curated signals than free-text description, so they're weighted higher.
- User vectors are built the same way from `preferred_categories` and `preferred_tags`.
- **Returning users** (non-empty `purchase_history`) get their vector blended: 60% average of their purchased products' vectors + 40% stated preferences. This lets past behavior dominate while still respecting newly stated interests.
- **Cold-start users** (empty `purchase_history`) use their stated-preference vector alone — there's no history to blend in, so the agent leans entirely on what they say they want. This is the graceful cold-start handling required by the brief: no purchases needed to get relevant results.

### Filtering vs. ranking
Two things are **hard filters** (never shown, regardless of similarity score):
- Products carrying any tag in `disliked_tags`.
- Products already in `purchase_history`.
- Products priced more than 15% over `budget_max` (a small buffer avoids losing a near-perfect match over a few dollars, while still respecting the budget in spirit).

Everything else is **ranked**, not filtered: cosine similarity score, plus a small `+0.05` bonus for an exact category match and `+0.03` for comfortably fitting the budget.

### Why not TF-IDF or a vector database?
A full TF-IDF/IDF weighting or embeddings-based approach would scale better to thousands of products, but for a 22-item demo catalogue, plain term-frequency vectors are simpler, fully explainable line-by-line, and require zero external dependencies — which matters for a reviewer running this from a README in a few minutes. See Tradeoffs below for what I'd change at scale.

### Where the LLM fits in
The brief's suggested architecture treats an LLM API as the agent's "brain." Here, the deterministic similarity engine is the brain for *ranking* (this needs to be reproducible and explainable, not creative). The LLM is used only for the *last-mile* task it's actually good at: turning a correct-but-dry rule-based reason into one natural sentence. If `ANTHROPIC_API_KEY` isn't set, or the API call fails for any reason (bad key, no network, rate limit), `llm_explainer.py` catches the exception and silently falls back to the rule-based reason — the agent never crashes or hangs because of the network.

---

## Sample Input & Output

Input (`data/users.json`, one entry):
```json
{
  "id": "user_004",
  "name": "Sneha Iyer",
  "preferred_categories": ["Fashion", "Beauty"],
  "preferred_tags": ["vegan", "cruelty-free", "sustainable"],
  "disliked_tags": ["leather"],
  "budget_max": 80,
  "purchase_history": []
}
```

Output (from `outputs/sample_output.txt`, generated by running `main.py`):
```
Recommendations for Sneha Iyer  (user_id=user_004)
COLD START (no purchase history)
========================================================================
1. Minimalist Leather-Free Sneakers ($69.99) [Fashion]
   -> Recommended (72.7% match): matches your preferred category 'Fashion';
      shares your interests: sustainable, vegan; fits within your $80 budget.
2. Natural Skincare Gift Set ($49.99) [Beauty]
   -> Recommended (60.9% match): matches your preferred category 'Beauty';
      shares your interests: cruelty-free, vegan; fits within your $80 budget.
3. Vegan Leather Wallet ($29.99) [Fashion]
   -> Recommended (47.9% match): matches your preferred category 'Fashion';
      shares your interests: vegan; fits within your $80 budget.
```

Notice the catalogue also contains a plain "Leather Wallet"-style item with the `leather` tag — it never appears here, because `disliked_tags` is a hard filter, not just a scoring penalty. Full output for all 4 sample users is in `outputs/sample_output.txt` after you run the agent once.

---

## Tradeoffs & Limitations

**What I optimized for:** an agent a reviewer can run in under 2 minutes with zero API keys, fully explainable line-by-line, that still demonstrates a real similarity method and graceful cold-start behavior.

**Known limitations and what I'd improve with more time:**
- **Vectorization is basic term-frequency, not TF-IDF.** Common words (e.g. "eco-friendly" appearing on many products) aren't down-weighted the way IDF would. At catalogue sizes beyond a few hundred items, I'd switch to `scikit-learn`'s `TfidfVectorizer` or embeddings (e.g. `sentence-transformers`) for better semantic matching (so "climbing gear" and "hiking equipment" would match even without exact word overlap).
- **No collaborative filtering.** Recommendations only look at a user's own stated interests/history, never "users like you also bought." Adding that would need more user-interaction data than a 4-profile demo has.
- **Cold-start relies entirely on explicit preferences.** A production system would also want implicit signals (browsing behavior, demographic priors) to recommend something reasonable even for a user who states nothing at all.
- **Budget buffer (15%) is a fixed heuristic.** A/B testing against real user behavior would tell us whether that's the right amount of flexibility, or whether it should vary by price bracket.
- **The optional LLM layer only rewrites text, it doesn't re-rank.** A more ambitious version could let the LLM re-rank or explain trade-offs between top candidates (e.g. "cheaper but fewer features" vs "pricier but perfect match") — I kept ranking deterministic on purpose so results are reproducible and testable.
- **No persistence layer.** User profiles and the catalogue are static JSON files. A real system would use SQLite (or a proper DB) so purchase history updates automatically as users buy things.

---

## Every part of this code, explained

- `src/recommender.py` — the entire recommendation engine: `tokenize()`, `Product.feature_vector()`, `UserProfile.stated_preference_vector()`, `cosine_similarity()`, and the `Recommender` class (`build_user_vector`, `recommend`, `_explain`). No dependencies beyond Python's standard library (`math`, `re`, `collections.Counter`).
- `src/llm_explainer.py` — one function, `enhance_reason_with_llm()`, that calls the Anthropic API and falls back safely on any failure.
- `src/main.py` — loads `data/*.json` into the dataclasses above, runs `recommend()` for each sample user, prints and saves results, and offers an interactive prompt.
- `tests/test_recommender.py` — four focused tests proving cold-start ranking, disliked-tag hard-filtering, purchase-history exclusion, and budget filtering all work as designed.
