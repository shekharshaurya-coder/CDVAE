"""
config.py — Central configuration for CDVAE backend
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class _Config:
    # ── MongoDB Atlas ──────────────────────────────────────────────────────────
    # Reads from .env → MONGO_URI
    MONGO_URI: str = os.getenv(
        "MONGO_URI",
        "mongodb://localhost:27017"  # fallback for local dev
    )

    # Database name inside the cluster
    MONGO_DB: str = os.getenv("MONGO_DB", "cdvae")

    # ── JWT ────────────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE_ME_IN_DOT_ENV")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_HOURS: int = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

    # ── Model ──────────────────────────────────────────────────────────────────
    CHECKPOINT_PATH: str = os.getenv(
        "CHECKPOINT_PATH",
        "cdvae_checkpoint_epoch_110.pt"
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    def validate(self):
        warnings = []

        if self.JWT_SECRET == "CHANGE_ME_IN_DOT_ENV":
            warnings.append("JWT_SECRET is not set — using insecure default.")

        if os.getenv("MONGO_URI") is None:
            warnings.append("MONGO_URI not set — using local MongoDB.")

        for w in warnings:
            print(f"[CONFIG WARN] {w}")


# Singleton instance
cfg = _Config()