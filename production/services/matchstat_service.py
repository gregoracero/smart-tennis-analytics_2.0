
import json
from pathlib import Path

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

CACHE_DIR = (
    ROOT
    / "cache"
    / "matchstat"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def load_cached_fixtures(
    tour,
    date
):

    cache_file = (
        CACHE_DIR
        / f"fixtures_{tour}_{date}.json"
    )

    if not cache_file.exists():

        raise FileNotFoundError(
            f"Fixture cache not found: {cache_file}"
        )

    with open(
        cache_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


if __name__ == "__main__":

    fixtures = load_cached_fixtures(
        "atp",
        "2026-08-16"
    )

    print(
        len(fixtures)
    )
