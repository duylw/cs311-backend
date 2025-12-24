from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
db_url = str(settings.DATABASE_URL)

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

if connect_args:
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        echo=settings.LOG_LEVEL == "DEBUG",
    )
else:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        echo=settings.LOG_LEVEL == "DEBUG",
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)