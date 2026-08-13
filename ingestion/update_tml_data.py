import datetime
from pathlib import Path

import requests

RAW_DIR = Path("data/raw/tml")

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TML_BASE = "https://stats.tennismylife.org/data"

HEADERS = {
    "User-Agent": "smart-tennis-analytics/1.0"
}


def download(filename):

    url = f"{TML_BASE}/{filename}"

    dest = RAW_DIR / filename

    print(f"Downloading {filename}...")

    r = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )

    r.raise_for_status()

    dest.write_bytes(
        r.content
    )

    print(
        f"OK {filename}"
    )


def main():

    year = datetime.date.today().year

    files = [
        f"{year}.csv",
        f"{year}_challenger.csv",
        "ongoing_tourneys.csv"
    ]

    print()
    print("=" * 60)
    print("TML UPDATE")
    print("=" * 60)

    for f in files:
        download(f)

    print()
    print("Done")


if __name__ == "__main__":
    main()

