"""Submit one bsub job per imputation, fold and seed."""
from common import submit_jobs

if __name__ == "__main__":
    submit_jobs("fibers")
