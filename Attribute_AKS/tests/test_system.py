"""
Test script for attribute extraction system
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import Config
from data_loader import VideoAVEAttrDataset
from aks_sampler import AdaptiveKeyframeSampler, SimpleTopKSampler, UniformSampler
from evaluation import fuzzy_f1_score, longest_common_substring_length


def test_data_loader():
    """Test dataset loading"""
    logger.info("Testing data loader...")
    
    try:
        dataset = VideoAVEAttrDataset(
            data_root=Config.DATASET_ROOT,
            domains=["beauty"],
            split="test",
            max_samples=10
        )
        
        logger.info(f"✓ Loaded {len(dataset)} samples")
        
        # Print sample
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"  Sample: {sample}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False


def test_aks_sampler():
    """Test AKS sampling"""
    logger.info("Testing AKS sampler...")
    
    try:
        # Create synthetic scores
        import numpy as np
        scores = [0.1, 0.2, 0.8, 0.9, 0.15, 0.3, 0.85, 0.2, 0.1, 0.05]
        M = 3
        
        # Test AKS
        sampler = AdaptiveKeyframeSampler(max_level=4, s_threshold=0.6)
        selected = sampler.select(scores, M)
        
        logger.info(f"✓ AKS selected {len(selected)} frames: {selected}")
        assert len(selected) == M, f"Expected {M} frames, got {len(selected)}"
        
        # Test baselines
        top_k_sampler = SimpleTopKSampler()
        top_k_selected = top_k_sampler.select(scores, M)
        logger.info(f"✓ Top-K selected: {top_k_selected}")
        
        uniform_sampler = UniformSampler()
        uniform_selected = uniform_sampler.select(scores, M)
        logger.info(f"✓ Uniform selected: {uniform_selected}")
        
        return True
    except Exception as e:
        logger.error(f"✗ AKS sampler test failed: {e}")
        return False


def test_fuzzy_f1():
    """Test Fuzzy F1 metric"""
    logger.info("Testing Fuzzy F1 metric...")
    
    try:
        # Test cases
        test_cases = [
            ("red", "red", 1.0),  # Exact match
            ("red", "Red", 1.0),  # Case insensitive
            ("red color", "red", 1.0),  # Substring match
            ("crimson", "red", 0.0),  # No match
            ("", "red", 0.0),  # Empty prediction
        ]
        
        for pred, label, expected in test_cases:
            score = fuzzy_f1_score(pred, label)
            status = "✓" if abs(score - expected) < 0.01 else "✗"
            logger.info(f"{status} fuzzy_f1('{pred}', '{label}') = {score:.2f}")
            assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"
        
        return True
    except Exception as e:
        logger.error(f"✗ Fuzzy F1 test failed: {e}")
        return False


def test_lcs():
    """Test longest common substring"""
    logger.info("Testing LCS...")
    
    try:
        test_cases = [
            ("hello", "hello", 5),      # Exact match
            ("hello", "hallo", 3),      # LCS: "llo"
            ("abc", "def", 0),          # No common substring
            ("", "test", 0),            # Empty string
            ("red", "red", 3),          # Exact match
            ("color red", "red", 3),    # LCS: "red"
        ]
        
        for s1, s2, expected in test_cases:
            lcs = longest_common_substring_length(s1, s2)
            status = "✓" if lcs == expected else "✗"
            logger.info(f"{status} lcs('{s1}', '{s2}') = {lcs}")
            assert lcs == expected, f"Expected {expected}, got {lcs}"
        
        return True
    except Exception as e:
        logger.error(f"✗ LCS test failed: {e}")
        return False


def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")
    
    try:
        logger.info(f"✓ QWEN_MODEL_PATH: {Config.QWEN_MODEL_PATH}")
        logger.info(f"✓ DATASET_ROOT: {Config.DATASET_ROOT}")
        logger.info(f"✓ M_ATTR: {Config.M_ATTR}")
        logger.info(f"✓ MAX_LEVEL: {Config.MAX_LEVEL}")
        logger.info(f"✓ S_THRESHOLD: {Config.S_THRESHOLD}")
        
        # Check if paths exist
        if os.path.exists(Config.DATASET_ROOT):
            logger.info(f"✓ Dataset directory exists")
        else:
            logger.warning(f"⚠ Dataset directory not found: {Config.DATASET_ROOT}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("SYSTEM TESTS")
    logger.info("="*60)
    
    tests = [
        # ("Configuration", test_config),
        # ("Data Loader", test_data_loader),
        ("AKS Sampler", test_aks_sampler),
        # ("Fuzzy F1", test_fuzzy_f1),
        # ("LCS", test_lcs),
    ]
    
    results = []
    for name, test_fn in tests:
        logger.info(f"\n--- {name} ---")
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
