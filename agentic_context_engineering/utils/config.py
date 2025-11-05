"""
Configuration management for ACE system.
Handles YAML config loading, environment variable overrides, and validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    model_config = ConfigDict(extra="allow")

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct"
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    gpu: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ACEConfig(BaseModel):
    """ACE iteration configuration."""

    model_config = ConfigDict(extra="allow")

    max_iterations: int = 10
    convergence_threshold: float = 0.05
    reflection_batch_size: int = 3
    early_stopping_patience: int = 2


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    model_config = ConfigDict(extra="allow")

    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge", "exact_match"])
    holdout_ratio: float = 0.2
    manual_review_samples: int = 10


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    model_config = ConfigDict(extra="allow")

    max_concurrent_requests: int = 1
    warm_start: bool = True
    log_vram_usage: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="allow")

    level: str = "INFO"
    save_intermediate: bool = True
    output_dir: str = "./outputs"
    structured_format: str = "json"


class ConfigManager:
    """Manages ACE configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/default.yaml"
        self._config = None

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable overrides.

        Args:
            config_path: Path to config file (optional)

        Returns:
            Configuration dictionary
        """
        if config_path:
            self.config_path = config_path

        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)

        # Validate configuration
        self._validate_config(config)

        self._config = config
        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        import os

        # LLM overrides
        if os.getenv("ACE_LLM_MODEL"):
            config["llm"]["model"] = os.getenv("ACE_LLM_MODEL")
        if os.getenv("ACE_LLM_BASE_URL"):
            config["llm"]["base_url"] = os.getenv("ACE_LLM_BASE_URL")
        if os.getenv("ACE_LLM_TEMPERATURE"):
            config["llm"]["parameters"]["temperature"] = float(os.getenv("ACE_LLM_TEMPERATURE"))

        # ACE overrides
        if os.getenv("ACE_MAX_ITERATIONS"):
            config["ace"]["max_iterations"] = int(os.getenv("ACE_MAX_ITERATIONS"))
        if os.getenv("ACE_CONVERGENCE_THRESHOLD"):
            config["ace"]["convergence_threshold"] = float(os.getenv("ACE_CONVERGENCE_THRESHOLD"))

        # Output directory override
        if os.getenv("ACE_OUTPUT_DIR"):
            config["logging"]["output_dir"] = os.getenv("ACE_OUTPUT_DIR")

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        required_sections = ["llm", "ace", "evaluation", "performance", "logging"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate LLM config
        llm_config = config["llm"]
        if "model" not in llm_config:
            raise ValueError("Missing 'model' in LLM config")
        if "base_url" not in llm_config:
            raise ValueError("Missing 'base_url' in LLM config")

        # Validate ACE config
        ace_config = config["ace"]
        if ace_config.get("max_iterations", 0) <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 <= ace_config.get("convergence_threshold", 0) <= 1:
            raise ValueError("convergence_threshold must be between 0 and 1")

        # Validate evaluation config
        eval_config = config["evaluation"]
        if not 0 <= eval_config.get("holdout_ratio", 0) <= 1:
            raise ValueError("holdout_ratio must be between 0 and 1")

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration as Pydantic model."""
        if not self._config:
            self.load_config()
        return LLMConfig(**self._config["llm"])

    def get_ace_config(self) -> ACEConfig:
        """Get ACE configuration as Pydantic model."""
        if not self._config:
            self.load_config()
        return ACEConfig(**self._config["ace"])

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration as Pydantic model."""
        if not self._config:
            self.load_config()
        return EvaluationConfig(**self._config["evaluation"])

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration as Pydantic model."""
        if not self._config:
            self.load_config()
        return PerformanceConfig(**self._config["performance"])

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration as Pydantic model."""
        if not self._config:
            self.load_config()
        return LoggingConfig(**self._config["logging"])

    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        if not self._config:
            raise ValueError("No configuration loaded")

        with open(output_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value."""
        if not self._config:
            self.load_config()

        if section not in self._config:
            self._config[section] = {}

        self._config[section][key] = value
        self._validate_config(self._config)

    @classmethod
    def create_default_config(cls, output_path: str) -> None:
        """Create a default configuration file."""
        default_config = {
            "llm": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "llama3.1:8b-instruct-fp16",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                    "repeat_penalty": 1.1,
                },
                "gpu": {"num_gpu": 1, "gpu_memory_fraction": 0.9, "enable_kv_cache": True},
            },
            "ace": {
                "max_iterations": 10,
                "convergence_threshold": 0.05,
                "reflection_batch_size": 3,
                "early_stopping_patience": 2,
            },
            "evaluation": {
                "metrics": ["bleu", "rouge", "exact_match", "semantic_similarity"],
                "holdout_ratio": 0.2,
                "manual_review_samples": 10,
            },
            "performance": {"max_concurrent_requests": 1, "warm_start": True, "log_vram_usage": True},
            "logging": {
                "level": "INFO",
                "save_intermediate": True,
                "output_dir": "./outputs",
                "structured_format": "json",
            },
        }

        with open(output_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
