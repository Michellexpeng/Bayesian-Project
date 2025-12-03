"""Update model metadata with test accuracy.

This script adds test accuracy to the model metadata by running the test scripts.

Usage:
  python scripts/update_model_accuracy.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import subprocess
import re


def extract_accuracy_from_output(output: str) -> float | None:
    """Extract accuracy percentage from test script output."""
    match = re.search(r'Test Accuracy:\s+([\d.]+)%', output)
    if match:
        return float(match.group(1))
    return None


def update_model_metadata(model_path: Path, accuracy: float):
    """Add test accuracy to model metadata."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if 'metadata' not in model:
        model['metadata'] = {}
    
    model['metadata']['test_accuracy'] = accuracy
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Updated {model_path.name}: test_accuracy = {accuracy:.2f}%")


def main():
    project_root = Path(__file__).resolve().parents[1]
    baseline_model = project_root / 'models' / 'hmm_baseline.pkl'
    conditional_model = project_root / 'models' / 'hmm_conditional.pkl'
    
    print("\n" + "=" * 80)
    print("Updating Model Metadata with Test Accuracy")
    print("=" * 80 + "\n")
    
    # Test baseline model
    print("[1/2] Testing baseline model...")
    result = subprocess.run(
        ['python', 'scripts/test_baseline.py', 
         '--model', str(baseline_model),
         '--pop909', 'data/POP909'],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    baseline_acc = extract_accuracy_from_output(result.stdout)
    if baseline_acc:
        update_model_metadata(baseline_model, baseline_acc)
    else:
        print("✗ Could not extract baseline accuracy")
    
    # Test conditional model
    print("\n[2/2] Testing conditional model...")
    result = subprocess.run(
        ['python', 'scripts/test_conditional.py',
         '--model', str(conditional_model),
         '--pop909', 'data/POP909'],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    conditional_acc = extract_accuracy_from_output(result.stdout)
    if conditional_acc:
        update_model_metadata(conditional_model, conditional_acc)
    else:
        print("✗ Could not extract conditional accuracy")
    
    print("\n" + "=" * 80)
    print("✓ Model metadata updated successfully")
    print("=" * 80 + "\n")
    
    # Show comparison
    if baseline_acc and conditional_acc:
        improvement = (conditional_acc - baseline_acc) / baseline_acc * 100
        print(f"Accuracy Comparison:")
        print(f"  Baseline:    {baseline_acc:.2f}%")
        print(f"  Conditional: {conditional_acc:.2f}%")
        print(f"  Improvement: +{improvement:.1f}%\n")


if __name__ == "__main__":
    main()
