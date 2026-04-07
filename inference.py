"""Competition entrypoint wrapper.

This keeps the required `inference.py` at the repository root while delegating
to the maintained implementation in `rl_agent.inference`.
"""

from rl_agent.inference import main


if __name__ == "__main__":
    main()
