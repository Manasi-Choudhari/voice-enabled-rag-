"""Generation configuration (Groq)."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = 512
TEMPERATURE = 0.1
MAX_RETRIES = 2
