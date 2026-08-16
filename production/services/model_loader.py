
import json
import joblib
from pathlib import Path

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


def load_model(
    tour,
    surface,
    engine="xgboost",
    version="v1"
):

    model_key = (
        f"{tour}_{surface}_{engine}_{version}"
        .lower()
    )

    model_dir = (
        ROOT
        / "ml"
        / "models"
        / model_key
    )

    if not model_dir.exists():

        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    model = joblib.load(
        model_dir / "model.joblib"
    )

    imputer = joblib.load(
        model_dir / "imputer.joblib"
    )

    with open(
        model_dir / "features.json",
        "r",
        encoding="utf-8"
    ) as f:

        features = json.load(f)

    with open(
        model_dir / "metrics.json",
        "r",
        encoding="utf-8"
    ) as f:

        metrics = json.load(f)

    with open(
        model_dir / "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return {

        "model": model,

        "imputer": imputer,

        "features": features,

        "metrics": metrics,

        "metadata": metadata,

        "model_dir": str(model_dir)
    }


if __name__ == "__main__":

    assets = load_model(
        tour="atp",
        surface="hard"
    )

    print()

    print("MODEL LOADED")

    print(
        assets["metadata"]
    )

    print()

    print(
        assets["metrics"]
    )
