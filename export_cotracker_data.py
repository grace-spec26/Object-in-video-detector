from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "co-tracker-main"))

from cotracker_wound_export import main  # noqa: E402


if __name__ == "__main__":
    main()
