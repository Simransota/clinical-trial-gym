"""rxgym — Clinical Trial RL Environment."""

try:
    from .models import RlAgentAction, RlAgentObservation
    from .client import RlAgentEnv
    from .tasks import TASKS, TASK_IDS
except ImportError:
    from models import RlAgentAction, RlAgentObservation
    from client import RlAgentEnv
    from tasks import TASKS, TASK_IDS

__all__ = ["RlAgentAction", "RlAgentObservation", "RlAgentEnv", "TASKS", "TASK_IDS"]
__version__ = "1.0.0"
