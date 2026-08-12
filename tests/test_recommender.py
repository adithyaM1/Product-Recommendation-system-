"""
test_recommender.py
--------------------
Lightweight assert-based tests (no pytest required).

Run with:
    python tests/test_recommender.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recommender import Product, UserProfile, Recommender  # noqa: E402


def test_cold_start_returns_relevant_recommendation():
    products = [
        Product("p1", "Eco Backpack", "Outdoor", "BrandA", 50, ["eco-friendly", "hiking"], "A sustainable backpack for hikes."),
        Product("p2", "Gaming Mouse", "Electronics", "BrandB", 40, ["gaming", "tech"], "A precise mouse for gamers."),
    ]
    user = UserProfile(id="u1", name="Test User", preferred_categories=["Outdoor"], preferred_tags=["eco-friendly"], budget_max=100)
    rec = Recommender(products)
    results = rec.recommend(user, top_n=2)
    assert results[0]["product"].id == "p1", "Eco backpack should rank first for an eco-focused cold-start user"
    print("PASS: test_cold_start_returns_relevant_recommendation")


def test_disliked_tags_are_filtered_out():
    products = [
        Product("p1", "Leather Wallet", "Fashion", "BrandA", 30, ["leather", "wallet"], "A classic leather wallet."),
        Product("p2", "Vegan Wallet", "Fashion", "BrandB", 32, ["vegan", "wallet"], "A cruelty-free vegan wallet."),
    ]
    user = UserProfile(id="u2", name="Test User 2", preferred_categories=["Fashion"], preferred_tags=["wallet"], disliked_tags=["leather"])
    rec = Recommender(products)
    ids = [r["product"].id for r in rec.recommend(user, top_n=5)]
    assert "p1" not in ids, "Products carrying a disliked tag must never be recommended"
    assert "p2" in ids, "Non-conflicting products should still be recommended"
    print("PASS: test_disliked_tags_are_filtered_out")


def test_purchase_history_excluded_from_recommendations():
    products = [
        Product("p1", "Running Shoes", "Fitness", "BrandA", 60, ["fitness", "running"], "Shoes for running."),
        Product("p2", "Yoga Mat", "Fitness", "BrandB", 25, ["fitness", "yoga"], "A mat for yoga."),
    ]
    user = UserProfile(id="u3", name="Test User 3", preferred_categories=["Fitness"], preferred_tags=["fitness"], purchase_history=["p1"])
    rec = Recommender(products)
    ids = [r["product"].id for r in rec.recommend(user, top_n=5)]
    assert "p1" not in ids, "Already-purchased items should not be re-recommended"
    print("PASS: test_purchase_history_excluded_from_recommendations")


def test_budget_filter_excludes_far_over_budget_items():
    products = [
        Product("p1", "Budget Bottle", "Outdoor", "BrandA", 20, ["outdoor"], "A cheap water bottle."),
        Product("p2", "Luxury Tent", "Outdoor", "BrandB", 500, ["outdoor"], "An extremely expensive tent."),
    ]
    user = UserProfile(id="u4", name="Test User 4", preferred_categories=["Outdoor"], preferred_tags=["outdoor"], budget_max=30)
    rec = Recommender(products)
    ids = [r["product"].id for r in rec.recommend(user, top_n=5)]
    assert "p2" not in ids, "Items far above budget should be hard-filtered out"
    print("PASS: test_budget_filter_excludes_far_over_budget_items")


if __name__ == "__main__":
    test_cold_start_returns_relevant_recommendation()
    test_disliked_tags_are_filtered_out()
    test_purchase_history_excluded_from_recommendations()
    test_budget_filter_excludes_far_over_budget_items()
    print("\nAll tests passed!")
