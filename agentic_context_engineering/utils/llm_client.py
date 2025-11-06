"""
LLM Client for ACE - Llama 3.1 8B Instruct via Ollama
Provides unified interface for Generator, Reflector, and Curator agents.
"""

import importlib
import logging
import time
from typing import Any, Dict, Optional

import torch
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Resolve Ollama driver lazily to avoid hard import for type checker and CI
_NEW_OLLAMA = False


def _resolve_ollama_cls():
    global _NEW_OLLAMA
    try:
        mod = importlib.import_module("langchain_ollama")
        _NEW_OLLAMA = True
        return mod.OllamaLLM
    except Exception:  # pragma: no cover
        mod = importlib.import_module("langchain_community.llms")
        _NEW_OLLAMA = False
        return mod.Ollama


# Expose name expected by tests for patching
Ollama = _resolve_ollama_cls()


class LLMConfig(BaseSettings):
    """Configuration for Llama 3.1 8B local inference"""

    model_config = ConfigDict(extra="allow")

    provider: str = "ollama"
    model: str = "llama3.1:8b-instruct-fp16"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2000
    num_ctx: int = 8192


class LLMClient:
    """
    Unified LLM client for all ACE agents (Generator, Reflector, Curator).
    Uses Llama 3.1 8B Instruct via Ollama for zero-cost local inference.

    Hardware Requirements:
    - GPU: NVIDIA RTX 4070 Ti Super (16GB VRAM)
    - RAM: 32GB DDR5
    - VRAM Usage: ~9GB
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._verify_gpu()
        self.llm = self._initialize_llm()
        self._generation_count = 0
        self._total_tokens = 0
        logger.info(f"✓ LLM client initialized: {self.config.model}")

    def _verify_gpu(self):
        """Verify GPU availability and VRAM capacity"""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Llama 3.1 8B requires GPU.\n"
                "Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
            )

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"✓ GPU: {gpu_name}")
        logger.info(f"✓ VRAM: {vram_gb:.1f} GB")

        if vram_gb < 10:
            logger.warning(
                f"⚠ Only {vram_gb:.1f}GB VRAM available. Llama 3.1 8B FP16 needs ~9GB. May encounter OOM errors."
            )

    def _initialize_llm(self):
        """Initialize Ollama LLM using alias `Ollama` (patched in tests)."""
        if _NEW_OLLAMA:
            # New API prefers model_kwargs for context/predict settings
            return Ollama(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                model_kwargs={
                    "num_ctx": self.config.num_ctx,
                    "num_predict": self.config.max_tokens,
                    "repeat_penalty": 1.1,
                },
            )
        # Fallback legacy
        return Ollama(
            model=self.config.model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            num_ctx=self.config.num_ctx,
            repeat_penalty=1.1,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from prompt using Llama 3.1 8B.

        Args:
            prompt: Input prompt
            **kwargs: Override config parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        start_time = time.time()

        try:
            # Forward only explicitly provided overrides; tests patch `.invoke`
            # and assert call signatures without implicit defaults.
            params = dict(kwargs)

            # new/old drivers accept different params; pass via invoke where supported
            try:
                response = self.llm.invoke(prompt, **params)
            except TypeError:
                # For some drivers that don't accept kwargs, fall back to bare call
                response = self.llm.invoke(prompt)

            # Track statistics
            self._generation_count += 1
            self._total_tokens += len(response.split())  # Rough token estimate

            elapsed = time.time() - start_time
            logger.debug(f"Generated {len(response)} chars in {elapsed:.2f}s")

            # langchain-ollama may return plain str; ensure string
            return getattr(response, "content", response)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Override config parameters

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses

    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage statistics"""
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "utilization_pct": round((allocated / total) * 100, 1),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "generation_count": self._generation_count,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_generation": self._total_tokens / max(1, self._generation_count),
            "vram_usage": self.get_vram_usage(),
        }

    def health_check(self) -> bool:
        """Verify Ollama connection and model availability"""
        try:
            test_response = self.generate("Hello", num_predict=5)
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def reset_stats(self):
        """Reset generation statistics"""
        self._generation_count = 0
        self._total_tokens = 0

    def warm_up(self) -> None:
        """Warm up the model with a test generation"""
        logger.info("Warming up LLM...")
        try:
            self.generate("Test prompt for warm-up", num_predict=10)
            logger.info("✓ LLM warmed up successfully")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")


# Example usage in agents
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = LLMClient()

    # Test generation
    response = client.generate("Explain the ACE methodology in 2 sentences.")
    print(f"Response: {response}\n")

    # Check VRAM
    vram = client.get_vram_usage()
    print(f"VRAM: {vram['allocated_gb']}/{vram['total_gb']} GB ({vram['utilization_pct']}%)")

    # Check stats
    stats = client.get_stats()
    print(f"Stats: {stats}")
