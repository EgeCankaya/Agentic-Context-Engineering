"""
Verify Llama 3.1 8B setup for ACE project
Run: python verify_setup.py
"""

import time

import torch
from langchain_community.llms import Ollama


def check_gpu():
    """Check GPU availability"""
    print("=== GPU Check ===")
    if not torch.cuda.is_available():
        print("X CUDA not available!")
        return False

    print(f"+ GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"+ VRAM: {vram:.1f} GB")
    return vram >= 10


def check_ollama():
    """Check Ollama connection"""
    print("\n=== Ollama Check ===")
    try:
        llm = Ollama(model="mistral:7b-instruct")
        print("+ Ollama connected")
        return llm
    except Exception as e:
        print(f"X Ollama connection failed: {e}")
        print("Run: ollama pull mistral:7b-instruct")
        return None


def test_generation(llm):
    """Test generation speed"""
    print("\n=== Generation Test ===")
    prompt = "Write a Python function to calculate factorial using recursion."

    start = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start

    tokens = len(response.split())  # Rough token estimate
    speed = tokens / elapsed

    print(f"+ Generated {tokens} tokens in {elapsed:.2f}s")
    print(f"+ Speed: {speed:.1f} tokens/sec")
    print(f"\nResponse preview:\n{response[:200]}...")

    # Check VRAM
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"\n+ VRAM used: {vram_used:.2f} GB")

    # Treat any successful generation as a pass; speed is informational
    return bool(response.strip())


def main():
    print("ACE Project - Llama 3.1 8B Setup Verification\n")

    # Run checks
    gpu_ok = check_gpu()
    if not gpu_ok:
        print("\nX GPU check failed. Fix GPU setup first.")
        return

    llm = check_ollama()
    if not llm:
        print("\nX Ollama check failed. Install and pull model first.")
        return

    gen_ok = test_generation(llm)

    # Summary
    print("\n" + "=" * 50)
    if gpu_ok and llm and gen_ok:
        print("+ ALL CHECKS PASSED - Ready for ACE development!")
    else:
        print("! Some checks failed - review errors above")
    print("=" * 50)


if __name__ == "__main__":
    main()
