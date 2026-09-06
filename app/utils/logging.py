"""Shared logging setup (loguru)."""
from loguru import logger

logger.add (
    "logs/app.log",
    rotation="10MB",
    retention="10 days"
)
