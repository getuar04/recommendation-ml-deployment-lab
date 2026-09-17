from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import MAX_FUTURE_EVENT_SKEW_SECONDS

# Group B: a small allowance for legitimate distributed-system clock skew between the
# client/producer and this server -- an event timestamp further in the future than this is
# rejected rather than silently accepted (see validate_watch below). Value now centralized in
# app.core.config.MAX_FUTURE_EVENT_SKEW_SECONDS (env-var-overridable, matching every other
# comparable threshold's existing convention) -- kept as a module-level timedelta here purely
# so validate_watch's own comparison arithmetic stays unchanged.
MAX_FUTURE_CLOCK_SKEW = timedelta(seconds=MAX_FUTURE_EVENT_SKEW_SECONDS)

class EventType(str, Enum):
    VIDEO_IMPRESSION="VIDEO_IMPRESSION"; VIDEO_STARTED="VIDEO_STARTED"; VIDEO_WATCHED="VIDEO_WATCHED"
    VIDEO_COMPLETED="VIDEO_COMPLETED"; VIDEO_SKIPPED="VIDEO_SKIPPED"; VIDEO_REWATCHED="VIDEO_REWATCHED"
    CONTENT_LIKED="CONTENT_LIKED"; CONTENT_SHARED="CONTENT_SHARED"; CONTENT_FAVORITED="CONTENT_FAVORITED"
    CONTENT_COMMENTED="CONTENT_COMMENTED"; CREATOR_FOLLOWED="CREATOR_FOLLOWED"; CONTENT_NOT_INTERESTED="CONTENT_NOT_INTERESTED"
    LIVE_IMPRESSION="LIVE_IMPRESSION"; LIVE_JOINED="LIVE_JOINED"; LIVE_LEFT="LIVE_LEFT"; LIVE_WATCHED="LIVE_WATCHED"
    LIVE_LIKED="LIVE_LIKED"; LIVE_SHARED="LIVE_SHARED"; LIVE_COMMENTED="LIVE_COMMENTED"
    LIVE_GIFT_SENT="LIVE_GIFT_SENT"; LIVE_CREATOR_FOLLOWED="LIVE_CREATOR_FOLLOWED"; LIVE_NOT_INTERESTED="LIVE_NOT_INTERESTED"

class EventCreate(BaseModel):
    # Watch percentage is server-derived from watchTimeSeconds/contentDurationSeconds. Reject
    # unknown fields so a caller cannot silently send an unsupported `watchPercentage` and
    # mistakenly believe it affected the stored interaction.
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    # max_length values match the bound app/api/content_routes.py's ContentCreate and
    # app/schemas/recommendation_schemas.py's Candidate already apply to the same logical
    # fields (contentId/creatorId/eventId/userId=128, category=64) -- established convention.
    event_id: str = Field(alias="eventId", min_length=1, max_length=128)
    user_id: str = Field(alias="userId", min_length=1, max_length=128)
    content_id: str = Field(alias="contentId", min_length=1, max_length=128)
    creator_id: str = Field(alias="creatorId", min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    event_type: EventType = Field(alias="eventType")
    watch_time_seconds: float | None = Field(None, alias="watchTimeSeconds", ge=0)
    live_watch_time_seconds: float | None = Field(None, alias="liveWatchTimeSeconds", ge=0)
    content_duration_seconds: float | None = Field(None, alias="contentDurationSeconds", gt=0)
    liked: bool = False; shared: bool = False; favorited: bool = False; commented: bool = False
    creator_followed: bool = Field(False, alias="creatorFollowed")
    timestamp: datetime

    @model_validator(mode="after")
    def validate_watch(self):
        self.category = self.category.strip().upper()
        if not self.category: raise ValueError("category must not be empty")
        # Timestamp-hardening finding: silently assuming a naive (no timezone) timestamp is
        # UTC let a client that actually serialized its own LOCAL wall-clock time (a client-
        # side bug, or simply forgetting to attach an offset) get misdiagnosed as "sending a
        # future timestamp" -- confusing, and it means the same naive string is silently
        # reinterpreted differently depending on nothing this server can see. Timezone must
        # come from the timestamp itself, never inferred (region/locale/etc.) -- an aware
        # timestamp with any explicit offset (`Z`, `+02:00`, `-05:00`, ...) is unaffected by
        # this check and continues to be compared correctly in UTC below.
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must include timezone information (e.g. a 'Z' suffix or an explicit +HH:MM/-HH:MM offset)")
        if self.timestamp - datetime.now(timezone.utc) > MAX_FUTURE_CLOCK_SKEW:
            raise ValueError("timestamp must not be in the future")
        if self.live_watch_time_seconds is not None:
            self.watch_time_seconds = self.live_watch_time_seconds
        if self.event_type.value.startswith("LIVE_"):
            return self
        if self.watch_time_seconds is not None and self.content_duration_seconds is None:
            raise ValueError("contentDurationSeconds is required with watchTimeSeconds")
        if self.watch_time_seconds is not None and self.watch_time_seconds > self.content_duration_seconds * 3:
            raise ValueError("watchTimeSeconds unreasonably exceeds duration")
        return self

class EventResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    event_id: str = Field(alias="eventId"); stored: bool
    watch_percentage: float | None = Field(alias="watchPercentage")
