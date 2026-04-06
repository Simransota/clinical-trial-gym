"""rxgym — Clinical Trial RL Environment."""

try:
    from .models import RlAgentAction, RlAgentObservation
    from .client import RlAgentEnv
except ImportError:
    from models import RlAgentAction, RlAgentObservation
    from client import RlAgentEnv

__all__ = ["RlAgentAction", "RlAgentObservation", "RlAgentEnv"]
__version__ = "1.0.0"
