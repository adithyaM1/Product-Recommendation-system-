"""
main.py
-------
Run this to see the agent recommend products for all sample users, then
optionally try it yourself interactively.

Usage:
    python src/main.py
"""

import json
import os
import sys
from pathlib import Path

# Make sibling modules importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommender import Product, UserProfile, Recommender  # noqa: E402
from llm_explainer import enhance_reason_with_llm  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set another way

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_products() -> list:
    with open(DATA_DIR / "products.json") as f:
        raw = json.load(f)
    return [Product(**p) for p in raw]


def load_users() -> list:
    with open(DATA_DIR / "users.json") as f:
        raw = json.load(f)
    users = []
    for u in raw:
        clean = {k: v for k, v in u.items() if not k.startswith("_")}
        users.append(UserProfile(**clean))
    return users


def format_recommendations(recommender: Recommender, user: UserProfile, use_llm: bool, top_n: int = 5) -> list:
    lines = []
    header = f"\n{'=' * 72}\nRecommendations for {user.name}  (user_id={user.id})"
    status = "COLD START (no purchase history)" if not user.purchase_history else f"Purchase history: {', '.join(user.purchase_history)}"
    header += f"\n{status}\n{'=' * 72}"
    lines.append(header)

    recs = recommender.recommend(user, top_n=top_n)
    if not recs:
        lines.append("No products matched the filters (try relaxing budget or disliked tags).")
        return lines

    for i, r in enumerate(recs, 1):
        product = r["product"]
        reason = r["reason"]
        if use_llm:
            reason = enhance_reason_with_llm(user.name, product.name, reason)
        lines.append(f"{i}. {product.name} (${product.price:.2f}) [{product.category}]\n   -> {reason}")

    return lines


def run_interactive(recommender: Recommender, use_llm: bool) -> None:
    print("\n" + "=" * 72)
    print("Try it yourself! Leave a field blank to skip it.")
    name = input("Your name: ").strip() or "Guest"
    categories = [c.strip() for c in input("Preferred categories (comma-separated, e.g. Electronics,Outdoor): ").split(",") if c.strip()]
    tags = [t.strip() for t in input("Interests/tags (comma-separated, e.g. eco-friendly,fitness): ").split(",") if t.strip()]
    disliked = [d.strip() for d in input("Disliked tags (comma-separated, or blank): ").split(",") if d.strip()]
    budget_raw = input("Max budget in $ (or blank for no limit): ").strip()
    budget_max = float(budget_raw) if budget_raw else None

    user = UserProfile(
        id="interactive_user",
        name=name,
        preferred_categories=categories,
        preferred_tags=tags,
        disliked_tags=disliked,
        budget_max=budget_max,
    )
    for line in format_recommendations(recommender, user, use_llm):
        print(line)


def main() -> None:
    products = load_products()
    users = load_users()
    recommender = Recommender(products)

    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    if use_llm:
        print("ANTHROPIC_API_KEY detected: reasons will be rephrased by Claude.\n")
    else:
        print("No ANTHROPIC_API_KEY set: using rule-based reasons only (the agent works fine without one).\n")

    all_lines = []
    for user in users:
        lines = format_recommendations(recommender, user, use_llm)
        for line in lines:
            print(line)
        all_lines += lines

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "sample_output.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"\nFull output saved to {output_path}")

    print("\n" + "=" * 72)
    try:
        choice = input("Would you like to enter your own preferences? (y/n): ").strip().lower()
    except EOFError:
        choice = "n"  # non-interactive environments (e.g. automated grading)
    if choice == "y":
        run_interactive(recommender, use_llm)


if __name__ == "__main__":
    main()
