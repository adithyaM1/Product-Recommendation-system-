# Product Recommendation Agent

> **One-sentence job statement:**\
> My agent takes a user's stated preferences (and optional purchase
> history) and produces a ranked list of product recommendations, each
> with a plain-English reason.

This is a **content-based filtering** agent: it compares a product
catalogue against a user's stated interests (and past purchases, if any)
using cosine similarity over weighted bag-of-words vectors. The core
logic needs **no API key and no internet connection** --- an LLM
(Claude) is used only as an *optional* layer to rephrase the reasons in
friendlier language.

------------------------------------------------------------------------

## Demo Screenshots

### 1. Product Recommendation Input

The user can enter their name, preferred category, interests, disliked
tags, and maximum budget.

![Product Recommendation Input](screenshots/product-input.png)

### 2. Recommendation Results

The agent returns a ranked list of products with match scores, prices,
and plain-English explanations.

![Recommendation Results](screenshots/recommendations-output.png)

------------------------------------------------------------------------

## Project Structure

``` text
product-recommendation-agent/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── products.json
│   └── users.json
├── src/
│   ├── recommender.py
│   ├── llm_explainer.py
│   └── main.py
├── tests/
│   └── test_recommender.py
├── outputs/
│   └── sample_output.txt
└── screenshots/
    ├── product-input.png
    └── recommendations-output.png
```

------------------------------------------------------------------------

## 1. Install

Requires Python 3.9+.

``` bash
cd product-recommendation-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you skip `pip install`, the agent still works --- it just won't have
the `anthropic` or `python-dotenv` packages, and will use rule-based
reasons.

------------------------------------------------------------------------

## 2. Configure (Optional)

Only needed if you want Claude to rephrase recommendation reasons in a
friendlier voice.

``` bash
cp .env.example .env
# Then edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

No API key? No problem --- the agent runs entirely offline with
rule-based reasons.

------------------------------------------------------------------------

## 3. Run

``` bash
python3 src/main.py
```

This will:

1.  Load the product catalogue and all 4 sample user profiles.
2.  Print top-5 recommendations with reasons for each user.
3.  Save the full output to `outputs/sample_output.txt`.
4.  Ask whether you'd like to enter your own preferences interactively.

------------------------------------------------------------------------

## 4. Run the Tests

``` bash
python3 tests/test_recommender.py
```

The tests check:

-   Cold-start ranking
-   Disliked-tag filtering
-   Purchase-history exclusion
-   Budget filtering

------------------------------------------------------------------------

## Design & Approach

### Similarity Method

Each product and user is converted into a **weighted bag-of-words
vector** and compared using **cosine similarity**.

-   Product vectors use:
    -   Category ×3
    -   Tags ×2
    -   Tokenized description ×1
-   User vectors are built from preferred categories and preferred tags.
-   Returning users get a blended vector:
    -   60% purchased-product vectors
    -   40% stated preferences
-   Cold-start users with no purchase history use their stated
    preferences alone.

This provides a simple and explainable recommendation method without
requiring an external database or API.

### Filtering vs. Ranking

The following are hard filters:

-   Products containing any `disliked_tags`
-   Products already present in `purchase_history`
-   Products priced more than 15% above `budget_max`

Remaining products are ranked using:

-   Cosine similarity
-   `+0.05` bonus for an exact category match
-   `+0.03` bonus for comfortably fitting the budget

------------------------------------------------------------------------

## Why Not TF-IDF or a Vector Database?

For a 22-item demonstration catalogue, plain term-frequency vectors are:

-   Easy to understand
-   Fully explainable
-   Dependency-free
-   Fast to run
-   Easy for a reviewer to test

For a much larger catalogue, the system could be upgraded to TF-IDF,
embeddings, or a vector database.

------------------------------------------------------------------------

## Where the LLM Fits In

The deterministic similarity engine handles **ranking**, because
recommendation scores should be reproducible and testable.

Claude is used only for the final explanation layer:

> Deterministic engine → ranked products → optional Claude explanation

If the API key is missing, the API call fails, the network is
unavailable, or the API is rate-limited, the system falls back to
rule-based explanations instead of crashing.

------------------------------------------------------------------------

## Sample Input

Example user profile:

``` json
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

------------------------------------------------------------------------

## Sample Output

``` text
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

The catalogue also contains products carrying the `leather` tag. Because
`leather` is listed under `disliked_tags`, those products are removed by
the hard-filtering step and do not appear in the recommendations.

------------------------------------------------------------------------

## Key Features

-   Content-based product recommendation
-   Cosine similarity scoring
-   Weighted product and user vectors
-   Cold-start recommendation support
-   Purchase-history personalization
-   Disliked-tag hard filtering
-   Budget-aware filtering
-   Explainable recommendation reasons
-   Optional Claude integration
-   Offline fallback without an API key
-   Interactive user preference input
-   Automated tests
-   JSON-based product and user data

------------------------------------------------------------------------

## Tradeoffs & Limitations

### 1. Basic Term Frequency

The current implementation uses term-frequency vectors rather than
TF-IDF.

**Future improvement:** Use `TfidfVectorizer` or sentence embeddings for
better semantic matching.

### 2. No Collaborative Filtering

The current system only considers the individual user's preferences and
history.

**Future improvement:** Add collaborative filtering so users can benefit
from patterns such as "users like you also bought."

### 3. Cold Start

Cold-start users depend on explicit preferences.

**Future improvement:** Use browsing activity, clicks, product
popularity, or other implicit signals.

### 4. Fixed Budget Buffer

The 15% budget buffer is a simple heuristic.

**Future improvement:** Tune this value using real user behavior or A/B
testing.

### 5. LLM Does Not Re-rank

Claude only improves the explanation text.

**Future improvement:** Use an LLM for deeper comparison and trade-off
explanations while keeping the deterministic ranking as the primary
score.

### 6. No Persistent Database

The project currently stores profiles and products in JSON files.

**Future improvement:** Use SQLite or another database so purchase
history can be updated automatically.

------------------------------------------------------------------------

## Code Overview

### `src/recommender.py`

Contains the main recommendation engine:

-   `tokenize()`
-   `Product.feature_vector()`
-   `UserProfile.stated_preference_vector()`
-   `cosine_similarity()`
-   `Recommender.build_user_vector()`
-   `Recommender.recommend()`
-   `Recommender._explain()`

The module uses only Python standard-library functionality such as
`math`, `re`, and `collections.Counter`.

### `src/llm_explainer.py`

Contains the optional Claude integration.

It rewrites deterministic recommendation reasons into friendlier
natural-language explanations and safely falls back when the API is
unavailable.

### `src/main.py`

Responsible for:

-   Loading JSON data
-   Creating product and user objects
-   Running recommendations
-   Printing results
-   Saving `outputs/sample_output.txt`
-   Running the interactive preference flow

### `tests/test_recommender.py`

Contains focused tests for:

-   Cold-start ranking
-   Disliked-tag filtering
-   Purchase-history exclusion
-   Budget filtering

------------------------------------------------------------------------

## Example User Flow

``` text
User preferences
      ↓
Preferred category
      +
Interests
      +
Disliked tags
      +
Maximum budget
      ↓
Build user preference vector
      ↓
Apply hard filters
      ↓
Calculate cosine similarity
      ↓
Apply category + budget bonuses
      ↓
Sort products by score
      ↓
Generate recommendation reasons
      ↓
Top product recommendations
```

------------------------------------------------------------------------

## Future Improvements

Possible next versions could include:

1.  TF-IDF or embedding-based semantic search
2.  Collaborative filtering
3.  SQLite/PostgreSQL persistence
4.  User login and profiles
5.  Product images and richer product metadata
6.  Feedback collection such as likes/dislikes
7.  Recommendation history
8.  FastAPI backend
9.  React frontend
10. Vector database integration
11. LLM-powered comparison of top recommendations
12. Deployment using Docker and a cloud platform

------------------------------------------------------------------------

## Conclusion

This project demonstrates a practical, explainable product
recommendation agent that can operate without an API key or internet
connection.

The core recommendation engine remains deterministic and testable, while
the optional LLM layer improves the quality of the user-facing
explanations. This separation makes the system simple enough for a demo
while leaving a clear path toward a production-scale recommendation
platform.
