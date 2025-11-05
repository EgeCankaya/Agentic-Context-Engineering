"""
Agentic Context Engineering (ACE) - Evolving Contexts for Self-Improving Language Models

A framework that enables self-improvement of large language models through context evolution
rather than model fine-tuning. The system iteratively improves performance by running a
three-phase loop: Generator → Reflector → Curator.
"""

from .eval.educational_rubric import EducationalEvaluator
from .playbook_schema import Playbook
from .runners import ACERunner
from .utils.config import ConfigManager
from .utils.conversation_logger import ConversationLogger
from .utils.llm_client import LLMClient

__version__ = "0.0.1"
__author__ = "Egemen Çankaya"
__email__ = "egemencankaya14@gmail.com"

__all__ = [
    "ACERunner",
    "ConfigManager",
    "ConversationLogger",
    "EducationalEvaluator",
    "LLMClient",
    "Playbook",
]
