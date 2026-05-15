"""Download the MovieLens Latest Small dataset from GroupLens."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import urllib.request


URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_ROOT / "ml-latest-small.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, archive_path)
    with ZipFile(archive_path) as archive:
        archive.extractall(DATA_ROOT)
    print(f"MovieLens data ready at {DATA_ROOT / 'ml-latest-small'}")


if __name__ == "__main__":
    main()

