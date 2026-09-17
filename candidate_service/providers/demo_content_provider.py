"""Local, deterministic, no-network stand-in for content lookup. Reuses the existing shared
demo candidate pool (scripts.seed_demo_users.build_shared_candidate_pool) rather than
inventing a second catalog -- every content_id this package can return already exists in the
same demo ecosystem the rest of the project's Postman/demo flow uses."""
from __future__ import annotations

from candidate_service.domain.models import ContentItem
from scripts.seed_demo_users import build_shared_candidate_pool, stable_uuid

_POOL_BY_ID = {entry["contentId"]: entry for entry in build_shared_candidate_pool()}


def content_id_for_slug(slug: str) -> str:
    return stable_uuid("demo2-candidate-content", slug)


class DemoContentProvider:
    def get(self, content_id: str) -> ContentItem | None:
        entry = _POOL_BY_ID.get(content_id)
        if entry is None:
            return None
        return ContentItem(
            content_id=entry["contentId"], creator_id=entry["creatorId"], category=entry["category"],
            popularity_score=entry["popularityScore"], title=entry["title"], hashtags=entry["hashtags"],
            topics=entry["topics"], entities=entry["entities"], subgenres=entry["subgenres"],
        )
