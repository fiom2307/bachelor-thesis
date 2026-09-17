import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.spatial_ablation import run_channel_ablation


def main() -> None:
    run_channel_ablation()


if __name__ == "__main__":
    main()
