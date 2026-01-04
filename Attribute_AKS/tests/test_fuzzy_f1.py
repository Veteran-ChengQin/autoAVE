"""
Test script to evaluate results using the updated fuzzy matching logic.
This script reads the results from exp4_api_video_url_100 and computes metrics.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import evaluate_results


def load_results(results_path: str) -> list:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def main():
    # Path to results file
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results/results_test_beauty.json"
    )
    
    print(f"Loading results from: {results_path}")
    
    # Load results
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")
    
    # Evaluate with different thresholds
    # thresholds = [0.5, 0.6, 0.7, 0.8]
    thresholds = [0.5]
    
    print("\n" + "="*80)
    print("EVALUATING WITH UPDATED FUZZY MATCHING LOGIC")
    print("="*80)
    
    for threshold in thresholds:
        print(f"\n\n{'='*80}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*80}")
        
        # Evaluate results
        metrics = evaluate_results(results, threshold=threshold)
        
        # Save metrics to file
        metrics_output_path = os.path.join(
            os.path.dirname(results_path),
            f"metrics_exp1_local_key_frames_threshold_{threshold}.json"
        )
        
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_output_path}")


if __name__ == "__main__":
    main()
