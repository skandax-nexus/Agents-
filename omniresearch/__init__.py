"""OmniResearch: an offline-first, multi-agent research system for any domain."""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .agents.orchestrator import ResearchOrchestrator  # noqa: F401
from .cognition.solver import ProblemSolver  # noqa: F401
from .cognition import Experience, ThinkingAgent  # noqa: F401
