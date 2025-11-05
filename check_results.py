#!/usr/bin/env python3
"""Simple script to check ACE results"""

import json
import os


def main():
    # Find the latest results file
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        print("No outputs directory found")
        return

    result_files = [f for f in os.listdir(output_dir) if f.startswith("ace_results_")]
    if not result_files:
        print("No ACE results found")
        return

    latest_file = max(result_files)
    print(f"Checking results from: {latest_file}")

    # Load and display results
    with open(os.path.join(output_dir, latest_file)) as f:
        results = json.load(f)

    print("\n" + "=" * 50)
    print("ACE SYSTEM SUCCESSFULLY RUN!")
    print("=" * 50)

    print(f"+ Iterations completed: {len(results['iterations'])}")
    print(f"+ Final playbook version: {results['final_playbook']}")

    if results["iterations"]:
        iteration = results["iterations"][0]
        print(f"+ Tasks processed: {len(iteration['generated_outputs'])}")
        print(f"+ Reflections completed: {len(iteration['reflections'])}")

        # Show sample output
        if iteration["generated_outputs"]:
            print("\nSample Generated Output:")
            print("-" * 30)
            sample = iteration["generated_outputs"][0]
            print(sample[:300] + "..." if len(sample) > 300 else sample)

        # Show metrics
        if "metrics" in iteration:
            metrics = iteration["metrics"]
            print("\nPerformance Metrics:")
            print(f"   • Average reflection score: {metrics.get('avg_reflection_score', 'N/A')}")
            print(f"   • Code example rate: {metrics.get('code_example_rate', 'N/A')}")
            print(f"   • Citation rate: {metrics.get('citation_rate', 'N/A')}")

    print(f"\nResults saved to: {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
