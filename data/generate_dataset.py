"""
Standalone dataset generator script.
Generates corporate_portfolio.csv with 603 realistic Uzbek corporate borrowers.
"""

from pathlib import Path
import sys

# Ensure src is importable
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nazorat_ai.data.generator import generate_synthetic_portfolio


def main():
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "corporate_portfolio.csv"

    print(f"Generating 603 corporate borrowers...")
    df = generate_synthetic_portfolio(n_samples=603, top20_count=20, seed=42)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} records to {csv_path}")


if __name__ == "__main__":
    main()
