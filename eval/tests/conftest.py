"""Keep DeepEval fully local: no telemetry, no cloud, no update pings. Set before
any deepeval import (conftest is imported ahead of the test modules). GDPR-relevant —
nothing about the eval run leaves the machine."""

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

# Some agent surfaces (e.g. rate_extraction) import ``core.config`` at module load, whose
# Settings has four required fields. Stub the three non-secret ones with harmless dummies
# so those modules import in the eval venv — nothing here ever connects (no DB/SQS/S3 call
# is on the graded path). ANTHROPIC_API_KEY is deliberately NOT stubbed: the paid gates key
# their skip-on-no-key off it, so a placeholder would make them try to run against 401s.
os.environ.setdefault("DATABASE_URL", "postgresql://eval:eval@localhost/eval-not-connected")
os.environ.setdefault("AGENT_QUEUE_URL", "https://eval.invalid/queue-not-called")
os.environ.setdefault("S3_BUCKET", "eval-bucket-not-called")
