"""Load and manage extraction schemas for ADE."""
import json
from pathlib import Path
from typing import Optional, List

SCHEMA_DIR = Path(__file__).parent


def load_schema(schema_name: str) -> Optional[dict]:
    """Load a schema by name from the schemas directory."""
    schema_path = SCHEMA_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        return None
    with open(schema_path, 'r') as f:
        return json.load(f)


def list_schemas() -> List[str]:
    """List all available schema names."""
    return [p.stem for p in SCHEMA_DIR.glob("*.json")]
