#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.model_manager import get_model_manager


def main() -> None:
    manager = get_model_manager()
    created = manager.bootstrap_models(force=False)
    print("Generated checkpoints:")
    for item in created:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
