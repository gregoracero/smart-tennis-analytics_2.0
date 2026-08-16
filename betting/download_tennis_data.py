
from pathlib import Path
import requests

BASE_URL = "https://www.tennis-data.co.uk"

YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025
]

OUTPUT_DIR = Path(
    "data/raw/tennis_data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for year in YEARS:

    url = f"{BASE_URL}/{year}/{year}.xlsx"

    output_file = (
        OUTPUT_DIR /
        f"atp_{year}.xlsx"
    )

    print()
    print(f"Downloading {year}")

    try:

        response = requests.get(
            url,
            timeout=60
        )

        response.raise_for_status()

        with open(
            output_file,
            "wb"
        ) as f:

            f.write(
                response.content
            )

        print(
            f"Saved: {output_file}"
        )

    except Exception as e:

        print(
            f"Failed {year}: {e}"
        )

print()
print("Done.")
