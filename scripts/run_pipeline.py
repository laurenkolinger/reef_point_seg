#!/usr/bin/env python3
"""Launch the TCRMP CVR-CLIP Pipeline Orchestrator."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_orchestrator"))
from app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCRMP Pipeline Orchestrator")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    app = create_app()
    print(f"\n  TCRMP CVR-CLIP Pipeline Orchestrator")
    print(f"  http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
