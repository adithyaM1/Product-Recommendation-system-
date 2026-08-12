import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import unquote

# Make sibling modules importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommender import UserProfile, Recommender  # noqa: E402
from llm_explainer import enhance_reason_with_llm  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class RecommendationRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if path.startswith("/api/"):
            return ""
        if path in ["/", ""]:
            path = "/index.html"
        requested = unquote(path)
        target = FRONTEND_DIR / requested.lstrip("/")
        return str(target)

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self.send_error(404, "API endpoint not found")
            return
        super().do_GET()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/recommend":
            self.send_error(404, "API endpoint not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON payload")
            return

        user = self._build_user_profile(payload)
        recommendations = self.server.recommender.recommend(user, top_n=payload.get("top_n", 5))
        response = {
            "user": {
                "id": user.id,
                "name": user.name,
                "preferred_categories": user.preferred_categories,
                "preferred_tags": user.preferred_tags,
                "disliked_tags": user.disliked_tags,
                "budget_max": user.budget_max,
                "purchase_history": user.purchase_history,
            },
            "recommendations": [
                {
                    "id": r["product"].id,
                    "name": r["product"].name,
                    "category": r["product"].category,
                    "price": r["product"].price,
                    "tags": r["product"].tags,
                    "reason": r["reason"],
                    "score": r["score"],
                }
                for r in recommendations
            ],
        }

        body = json.dumps(response, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        # Avoid noisy access logs for normal browser activity.
        return

    def _build_user_profile(self, payload: dict) -> UserProfile:
        categories = [c.strip() for c in payload.get("preferred_categories", []) if isinstance(c, str) and c.strip()]
        tags = [t.strip() for t in payload.get("preferred_tags", []) if isinstance(t, str) and t.strip()]
        disliked = [d.strip() for d in payload.get("disliked_tags", []) if isinstance(d, str) and d.strip()]
        budget_max = payload.get("budget_max")
        if budget_max is not None:
            try:
                budget_max = float(budget_max)
            except (TypeError, ValueError):
                budget_max = None

        return UserProfile(
            id=payload.get("id", "web_user"),
            name=payload.get("name", "Guest"),
            preferred_categories=categories,
            preferred_tags=tags,
            disliked_tags=disliked,
            budget_max=budget_max,
            purchase_history=payload.get("purchase_history", []),
        )


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def load_products() -> list:
    with open(DATA_DIR / "products.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    products = load_products()
    recommender = Recommender([UserProfile(**{}) for _ in []]) if False else None  # type: ignore
    # Use explicit product loading logic from recommender module to preserve structure.
    from recommender import Product  # noqa: F401
    product_objs = [Product(**p) for p in products]
    server_address = (host, port)
    RecommendationRequestHandler.protocol_version = "HTTP/1.1"
    httpd = ThreadedHTTPServer(server_address, RecommendationRequestHandler)
    httpd.recommender = Recommender(product_objs)

    print(f"Serving frontend on http://{host}:{port} (press Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()


def main() -> None:
    host = os.getenv("RECOMMENDER_HOST", DEFAULT_HOST)
    port = int(os.getenv("RECOMMENDER_PORT", DEFAULT_PORT))
    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
