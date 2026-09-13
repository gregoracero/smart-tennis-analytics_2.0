#!/usr/bin/env python3
"""Calibradores probabilísticos compartidos por training e inferencia."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling sobre el logit de una probabilidad binaria.

    Esta implementación debe permanecer estable porque los calibradores
    guardados con joblib hacen referencia a esta clase durante la carga.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
        )

    @staticmethod
    def _to_logit(
        probability: Any,
    ) -> np.ndarray:
        values = np.asarray(
            probability,
            dtype=float,
        ).reshape(-1)

        values = np.clip(
            values,
            1e-6,
            1.0 - 1e-6,
        )

        return np.log(
            values
            / (
                1.0
                - values
            )
        ).reshape(-1, 1)

    def fit(
        self,
        probability: Any,
        y_true: Any,
    ) -> "PlattCalibrator":
        target = np.asarray(
            y_true,
            dtype=np.int8,
        ).reshape(-1)

        self.model.fit(
            self._to_logit(
                probability,
            ),
            target,
        )

        return self

    def predict(
        self,
        probability: Any,
    ) -> np.ndarray:
        return self.model.predict_proba(
            self._to_logit(
                probability,
            )
        )[:, 1]

    def predict_proba(
        self,
        probability: Any,
    ) -> np.ndarray:
        """Compatibilidad con código de inferencia anterior.

        Devuelve únicamente la probabilidad de la clase positiva.
        """

        return self.predict(
            probability,
        )