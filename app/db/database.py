from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import DATABASE_URL


class Base(DeclarativeBase):
    pass

def make_engine(url: str = DATABASE_URL):
    options: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool
    return create_engine(url, **options)

engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
