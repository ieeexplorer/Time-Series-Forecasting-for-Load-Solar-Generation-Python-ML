"""Run a configured experiment directly from a repository checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from energy_forecasting.experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
