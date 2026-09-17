"""Proves the SOCIAL/COLLABORATIVE candidate flow end to end, locally:

    demo Follow/UserBehavior providers (candidate_service, no network)
        -> SOCIAL/COLLABORATIVE candidates
        -> existing, unmodified POST /api/v1/recommendation-ml-service/recommendations
        -> LogisticRegression + social reranking
        -> Top-N

Never trains, promotes, or resets anything -- reads the already-active model through the real
endpoint exactly like any other caller. If no local server is reachable, prints the generated
candidates only (still useful to inspect candidate_service's own output in isolation).

Usage:
    python -m scripts.run_social_candidate_demo --base-url http://localhost:3500
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from candidate_service.providers.demo_follow_provider import USER_A
from candidate_service.services.candidate_service import (
    generate_candidates,
    to_rms_payload,
)


def _post(base_url: str, path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    except urllib.error.URLError:
        return 0, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:3500")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    candidates = generate_candidates(USER_A, limit=args.limit)
    print(f"generated {len(candidates)} SOCIAL/COLLABORATIVE candidates for target user {USER_A}\n")
    for candidate in candidates:
        print(f"  source={candidate.source:14s} sourceUser={candidate.source_user_id[:8]}  "
              f"content={candidate.content.content_id[:8]} ({candidate.content.category})  "
              f"interestSimilarity={candidate.interest_similarity:.4f}  "
              f"relationshipStrength={candidate.relationship_strength:.4f}  "
              f"sourceUserEngagement={candidate.source_user_engagement:.4f}  "
              f"mutualFollow={candidate.mutual_follow}")

    payload = {"userId": USER_A, "limit": args.limit, "candidates": [to_rms_payload(c) for c in candidates]}
    print(f"\nposting {len(candidates)} candidates to {args.base_url}/api/v1/recommendation-ml-service/recommendations ...")
    status, body = _post(args.base_url, "/api/v1/recommendation-ml-service/recommendations", payload)
    if status == 0:
        print("no local server reachable -- candidate generation above is still valid, skipping the RMS call.")
        return
    if status != 200:
        print(f"RMS call failed: HTTP {status} {body}")
        return

    print(f"\nmodelVersion={body['modelVersion']}  strategy={body['strategy']}  interactionCount={body['interactionCount']}\n")
    by_id = {c.content.content_id: c for c in candidates}
    for rec in body["recommendations"]:
        social = by_id.get(rec["contentId"])
        provenance = f"  <- {social.source} via {social.source_user_id[:8]}" if social else ""
        print(f"  rank={rec['rank']:>2} score={rec['score']:.4f} reason={rec['reason']:<20s} {rec['category']:8s}{provenance}")


if __name__ == "__main__":
    main()
