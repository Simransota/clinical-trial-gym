"""Repository-root server wrapper for OpenEnv validators."""

import argparse

from rl_agent.server.app import app as app
from rl_agent.server.app import main as _main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    return _main(host=args.host, port=args.port)


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
