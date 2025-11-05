"""
Tests for LLM client functionality.
"""

from unittest.mock import Mock, patch

import pytest

from agentic_context_engineering.utils.llm_client import LLMClient, LLMConfig


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.provider == "ollama"
        assert config.model == "llama3.1:8b-instruct-fp16"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000

    def test_config_override(self):
        """Test configuration override."""
        config = LLMConfig(model="test-model", temperature=0.5, max_tokens=1000)

        assert config.model == "test-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000


class TestLLMClient:
    """Test LLM client functionality."""

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_llm_client_initialization(self, mock_ollama, mock_torch):
        """Test LLM client initialization."""
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070 Ti SUPER"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm

        # Initialize client
        config = LLMConfig()
        client = LLMClient(config)

        assert client.config == config
        assert client.llm == mock_llm
        mock_ollama.assert_called_once()

    @patch("agentic_context_engineering.utils.llm_client.torch")
    def test_gpu_verification_no_cuda(self, mock_torch):
        """Test GPU verification when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        with pytest.raises(RuntimeError, match="CUDA not available"):
            LLMClient()

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_generate(self, mock_ollama, mock_torch):
        """Test text generation."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Generated response"
        mock_ollama.return_value = mock_llm

        # Test generation
        client = LLMClient()
        response = client.generate("Test prompt")

        assert response == "Generated response"
        mock_llm.invoke.assert_called_once_with("Test prompt")

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_generate_batch(self, mock_ollama, mock_torch):
        """Test batch generation."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama
        mock_llm = Mock()
        mock_llm.invoke.side_effect = ["Response 1", "Response 2"]
        mock_ollama.return_value = mock_llm

        # Test batch generation
        client = LLMClient()
        responses = client.generate_batch(["Prompt 1", "Prompt 2"])

        assert responses == ["Response 1", "Response 2"]
        assert mock_llm.invoke.call_count == 2

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_vram_usage(self, mock_ollama, mock_torch):
        """Test VRAM usage monitoring."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9
        mock_torch.cuda.memory_allocated.return_value = 8e9
        mock_torch.cuda.memory_reserved.return_value = 9e9

        # Mock Ollama
        mock_ollama.return_value = Mock()

        # Test VRAM usage
        client = LLMClient()
        vram_usage = client.get_vram_usage()

        assert "allocated_gb" in vram_usage
        assert "reserved_gb" in vram_usage
        assert "total_gb" in vram_usage
        assert "utilization_pct" in vram_usage
        assert vram_usage["total_gb"] == 16.0

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_health_check(self, mock_ollama, mock_torch):
        """Test health check functionality."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response"
        mock_ollama.return_value = mock_llm

        # Test health check
        client = LLMClient()
        is_healthy = client.health_check()

        assert is_healthy is True
        mock_llm.invoke.assert_called_once_with("Hello", num_predict=5)

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_health_check_failure(self, mock_ollama, mock_torch):
        """Test health check failure."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Connection failed")
        mock_ollama.return_value = mock_llm

        # Test health check failure
        client = LLMClient()
        is_healthy = client.health_check()

        assert is_healthy is False

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_stats(self, mock_ollama, mock_torch):
        """Test statistics tracking."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9
        mock_torch.cuda.memory_allocated.return_value = 8e9

        # Mock Ollama
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response with multiple words"
        mock_ollama.return_value = mock_llm

        # Test stats
        client = LLMClient()
        client.generate("Test prompt")
        stats = client.get_stats()

        assert stats["generation_count"] == 1
        assert stats["total_tokens"] > 0
        assert "vram_usage" in stats

    @patch("agentic_context_engineering.utils.llm_client.torch")
    @patch("agentic_context_engineering.utils.llm_client.Ollama")
    def test_reset_stats(self, mock_ollama, mock_torch):
        """Test statistics reset."""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 16e9

        # Mock Ollama
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response"
        mock_ollama.return_value = mock_llm

        # Test reset
        client = LLMClient()
        client.generate("Test prompt")
        client.reset_stats()
        stats = client.get_stats()

        assert stats["generation_count"] == 0
        assert stats["total_tokens"] == 0
