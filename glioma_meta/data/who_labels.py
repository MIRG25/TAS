"""Versioned WHO endpoints: never infer the endpoint from binary labels."""
import pandas as pd

LEGACY_WHO = "0=grade2/3,1=grade4"
LOW_HIGH_WHO = "0=grade1/2,1=grade3/4"
WHO_MAPS = {
    LEGACY_WHO: {2: 0, 3: 0, 4: 1},
    LOW_HIGH_WHO: {1: 0, 2: 0, 3: 1, 4: 1},
}


def map_who_grades(grades: pd.Series, definition: str = LOW_HIGH_WHO) -> pd.Series:
    if definition not in WHO_MAPS:
        raise ValueError(f"Unknown WHO label definition: {definition}")
    return pd.to_numeric(grades, errors="coerce").map(WHO_MAPS[definition])


def manifest_who_definition(manifest: pd.DataFrame) -> str:
    # Historical manifests have no marker. Preserve their documented endpoint;
    # never silently reinterpret them as the new low/high-grade task.
    if "who_label_definition" not in manifest:
        return LEGACY_WHO
    values = manifest["who_label_definition"]
    if values.isna().any() or values.nunique() != 1 or values.iloc[0] not in WHO_MAPS:
        raise ValueError("Manifest has missing, mixed or unknown WHO label definitions")
    return str(values.iloc[0])
