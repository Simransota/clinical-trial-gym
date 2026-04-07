"""Repository-root server wrapper for OpenEnv validators."""

from rl_agent.server.app import app as app
from rl_agent.server.app import main as _main


def main(host: str = "0.0.0.0", port: int = 8000):
    return _main(host=host, port=port)


__all__ = ["app", "main"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
