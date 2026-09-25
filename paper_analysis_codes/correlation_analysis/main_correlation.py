"""Submit one bsub job per fold; reuse its correlations across all seeds."""
from common import submit_jobs

if __name__ == "__main__":
    submit_jobs("correlation")
