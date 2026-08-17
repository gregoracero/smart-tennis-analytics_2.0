from pathlib import Path
import requests

BASE_URL = (
    "https://www.tennis-data.co.uk"
)

YEARS = list(
    range(2010, 2027)
)

OUTPUT_DIR = Path(
    "data/raw/tennis_data_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for year in YEARS:

    url = (
        f"{BASE_URL}/{year}/{year}.zip"
    )

    target = (
        OUTPUT_DIR /
        f"{year}.zip"
    )

    print()
    print(f"Downloading {year}")

    try:

        response = requests.get(
            url,
            timeout=120
        )

        response.raise_for_status()

        target.write_bytes(
            response.content
        )

        print(
            f"Saved: {target}"
        )

    except Exception as e:

        print(
            f"Failed {year}: {e}"
        )

print()
print("DONE")