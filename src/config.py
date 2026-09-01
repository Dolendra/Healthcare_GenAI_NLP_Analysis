"""Central configuration for the Healthcare GenAI NLP pipeline."""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ── Student / submission ID ────────────────────────────────────────────────
STUDENT_ID = "B210212"

# ── Input filenames (exact names required by assignment) ───────────────────
MEDIA_FILE = DATA_DIR / "Media & Research Articles data.xlsx"
TWITTER_FILE = DATA_DIR / "Twitter Posts Data.xlsx"

# ── Output filenames ───────────────────────────────────────────────────────
PREDICTIONS_FILE = OUTPUT_DIR / f"predictions_{STUDENT_ID}.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
STANDARDIZED_FILE = OUTPUT_DIR / "standardized_dataset.csv"

# ── Gemini model ───────────────────────────────────────────────────────────
# Primary Gemini model used for this assignment.
GEMINI_MODEL = "gemini-3.5-flash-lite"

# ── Processing settings ────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff
CHECKPOINT_INTERVAL = 10  # save every N records (100-record dataset)
REQUEST_DELAY = 4.5  # seconds — free tier ~15 req/min for gemini-3.5-flash-lite
PILOT_SAMPLE_SIZE = 10
VALIDATION_SAMPLE_SIZE = 50
EXPECTED_MEDIA_ROWS = 50
EXPECTED_TWITTER_ROWS = 50
EXPECTED_COMBINED_ROWS = 100

# Assignment-required text field vs NLP input with Twitter reply context
ASSIGNMENT_TEXT_COLUMN = "Combined"
NLP_CONTEXT_COLUMN = "Contextual_Text"

# ── Target schema after standardization ────────────────────────────────────
TARGET_COLUMNS = [
    "Record_ID",
    "unique_id",
    "Title",
    "Body",
    "Combined",
    "Contextual_Text",
    "Source",
    "Text_Type",
    "Original_Source",
    "Source Type",
    "Link",
    "Published date",
    "HCP Handle",
    "replied_to_tweet",
]

# ── Allowed NLP values ─────────────────────────────────────────────────────
TOPICS = [
    "Efficacy-General",
    "Progression Free Survival (PFS)",
    "Overall Survival (OS)",
    "Safety-General",
    "Safety-Side Effects",
    "General Opinion",
    "Others",
]

SENTIMENTS = ["positive", "negative", "neutral"]

VALID_TOPICS = set(TOPICS)
VALID_SENTIMENTS = set(SENTIMENTS)

# Real assignment column maps.
# Do NOT map Source → Source. Preserve publisher/platform as Original_Source.
MEDIA_COLUMN_MAP = {
    "unique_id": "unique_id",
    "Article title": "Title",
    "Content": "Body",
    "Source": "Original_Source",
    "Source Type": "Source Type",
    "Article link": "Link",
    "Published date": "Published date",
}

TWITTER_COLUMN_MAP = {
    "unique_id": "unique_id",
    "HCP Handle": "HCP Handle",
    "Posts": "Body",
    "replied_to_tweet": "replied_to_tweet",
    "Source": "Original_Source",
    "Source Type": "Source Type",
    "Post link": "Link",
    "Published date": "Published date",
}
