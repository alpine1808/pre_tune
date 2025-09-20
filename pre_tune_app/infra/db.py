from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from pre_tune_app.config.settings import AppConfig

def build_engine(cfg: AppConfig) -> Engine:
    return create_engine(cfg.db_url, pool_pre_ping=True, pool_recycle=1800)
