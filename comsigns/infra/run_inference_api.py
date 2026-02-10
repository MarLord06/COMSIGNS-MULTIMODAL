#!/usr/bin/env python3
"""
Start the ComSigns Inference API server.

Usage:
    python run_api.py [--port PORT] [--host HOST] [--reload]
"""

import sys
from pathlib import Path

# Add comsigns to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="ComSigns Inference API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"Starting ComSigns API on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
