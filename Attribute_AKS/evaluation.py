"""
Evaluation metrics for attribute value extraction
"""
import logging
import os
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def custom_fuzzy_match(label: str, pred: str, threshold: float = 0.5) -> bool:
    """
    Custom fuzzy match based on common substring rule.
    
    A match is successful if the longest common substring between label and pred
    is >= threshold * len(label).
    
    Args:
        label: Ground truth value
        pred: Predicted value
        threshold: Matching threshold (default 0.5)
        
    Returns:
        True if match, False otherwise
    """
    label = str(label).lower().strip()
    pred = str(pred).lower().strip()

    if not label or not pred:
        return False

    m, n = len(label), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if label[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])

    return max_len >= threshold * len(label)



def compute_fuzzy_f1_scores(predictions: List, labels: List, 
                            attr_names: List[str] = None,
                            threshold: float = 0.5) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Compute Fuzzy F1 scores at both overall and attribute levels.
    
    Handles predictions as dictionaries (e.g., {'Color': 'black'})
    Based on TP/FP/FN statistics with custom fuzzy matching.
    
    Args:
        predictions: List of predicted values (can be dict or str)
        labels: List of ground truth values (strings)
        attr_names: List of attribute names (for per-attribute metrics)
        threshold: Matching threshold (default 0.5)
        
    Returns:
        Tuple of (precision, recall, f1, attr_f1_scores)
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    attr_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        attr_name = attr_names[i] if attr_names else f"attr_{i}"
        
        # Extract predicted value from dict or use as string
        if isinstance(pred, dict):
            # pred is a dict like {'Color': 'black'}
            # Extract the value for the current attribute
            pred_str = pred.get(attr_name, "").strip() if pred else ""
        else:
            # pred is a string
            pred_str = str(pred).strip()
        
        label_str = str(label).strip()
        
        # Handle empty predictions (MLLM failed to extract)
        if not pred_str:
            # False negative: attribute not extracted
            total_fn += 1
            attr_stats[attr_name]["fn"] += 1
        elif not label_str:
            # False positive: predicted something when ground truth is empty
            total_fp += 1
            attr_stats[attr_name]["fp"] += 1
        else:
            # Both pred and label are non-empty
            if custom_fuzzy_match(label_str, pred_str, threshold):
                # True positive: correct extraction
                total_tp += 1
                attr_stats[attr_name]["tp"] += 1
            else:
                # False negative: incorrect extraction
                total_fn += 1
                attr_stats[attr_name]["fn"] += 1
    
    # Overall scores
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Attribute-level scores
    attr_f1_scores = {}
    for attr, stats in attr_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_attr = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        attr_f1_scores[attr] = round(f1_attr, 4)
    
    return round(precision, 4), round(recall, 4), round(f1, 4), attr_f1_scores


def compute_per_product_f1_scores(results: List[Dict], threshold: float = 0.5) -> Tuple[Dict, Dict]:
    """
    Compute F1 scores aggregated by product_id (严谨的评估方式).
    
    This is the rigorous evaluation method: for each product, we aggregate all its attributes
    into a single ground truth dict and prediction dict, then compute F1 scores.
    
    Args:
        results: List of result dictionaries from inference, each with format:
                 {
                     'product_id': str,
                     'category': str,
                     'attr_name': str,
                     'pred': dict like {'Color': 'black'},
                     'label': str like 'White',
                     'num_keyframes': int,
                     'keyframe_indices': list,
                     ...
                 }
        threshold: Matching threshold for fuzzy matching (default 0.5)
        
    Returns:
        Tuple of (per_product_metrics, overall_metrics)
        - per_product_metrics: Dict[product_id] -> {precision, recall, f1, tp, fp, fn, attr_count}
        - overall_metrics: {precision, recall, f1, tp, fp, fn, total_products, total_attributes}
    """
    # Group results by product_id
    products = defaultdict(list)
    for result in results:
        if "error" not in result:
            product_id = result.get("product_id", "unknown")
            products[product_id].append(result)
    
    # Compute F1 for each product
    per_product_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    total_products = 0
    total_attributes = 0
    
    for product_id, product_results in products.items():
        # Build ground truth dict and prediction dict for this product
        gt_dict = {}  # {attr_name: label_value}
        pred_dict = {}  # {attr_name: predicted_value}
        
        try:
            for result in product_results:
                attr_name = result.get("attr_name", "")
                label = result.get("label", "")
                pred = result.get("pred", {})
                
                # Extract predicted value from dict
                if isinstance(pred, dict):
                    pred_value = pred.get(attr_name, "").strip()
                else:
                    pred_value = str(pred).strip()
                
                gt_dict[attr_name] = str(label).strip()
                pred_dict[attr_name] = pred_value
        except Exception as e:
            print(f"Warning: Skipping product {product_id} due to key extraction error: {e}")
            continue
        
        # Compute TP/FP/FN for this product (following the evaluation.ipynb logic)
        product_tp, product_fp, product_fn = 0, 0, 0
        matched_keys = set()
        
        # Check ground truth attributes for Recall
        for attr_name, gt_value in gt_dict.items():
            if attr_name in pred_dict:
                pred_value = pred_dict[attr_name]
                
                if custom_fuzzy_match(gt_value, pred_value, threshold):
                    # Correct match
                    product_tp += 1
                    matched_keys.add(attr_name)
                else:
                    # Incorrect match
                    product_fn += 1
                
            else:
                # Attribute not predicted 漏报 
                product_fn += 1
        
        # Check predicted attributes not in ground truth for Precise
        for attr_name, pred_value in pred_dict.items():
            if attr_name not in gt_dict:
                product_fp += 1
            elif attr_name not in matched_keys :
                # Attribute predicted but not in ground truth 误报 (false positive)
                product_fp += 1
        
        # Compute metrics for this product
        product_precision = product_tp / (product_tp + product_fp) if (product_tp + product_fp) > 0 else 0.0
        product_recall = product_tp / (product_tp + product_fn) if (product_tp + product_fn) > 0 else 0.0
        product_f1 = 2 * product_precision * product_recall / (product_precision + product_recall) \
                     if (product_precision + product_recall) > 0 else 0.0
        
        per_product_metrics[product_id] = {
            "precision": round(product_precision, 4),
            "recall": round(product_recall, 4),
            "f1": round(product_f1, 4),
            "tp": product_tp,
            "fp": product_fp,
            "fn": product_fn,
            "attr_count": len(gt_dict),
        }
        
        # Accumulate for overall metrics
        total_tp += product_tp
        total_fp += product_fp
        total_fn += product_fn
        total_products += 1
        total_attributes += len(gt_dict)
    
    # Compute overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0.0
    
    overall_metrics = {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_products": total_products,
        "total_attributes": total_attributes,
    }
    
    return per_product_metrics, overall_metrics


def compute_per_category_f1_scores(results: List[Dict], threshold: float = 0.5) -> Tuple[Dict, Dict]:
    """
    Compute F1 scores aggregated by category.
    
    For each category, we aggregate all products and their attributes,
    then compute F1 scores at the category level.
    
    Args:
        results: List of result dictionaries from inference, each with format:
                 {
                     'product_id': str,
                     'category': str,
                     'attr_name': str,
                     'pred': dict like {'Color': 'black'},
                     'label': str like 'White',
                     'num_keyframes': int,
                     'keyframe_indices': list,
                     ...
                 }
        threshold: Matching threshold for fuzzy matching (default 0.5)
        
    Returns:
        Tuple of (per_category_metrics, overall_metrics)
        - per_category_metrics: Dict[category] -> {precision, recall, f1, tp, fp, fn, product_count, attr_count}
        - overall_metrics: {precision, recall, f1, tp, fp, fn, total_categories, total_products, total_attributes}
    """
    # Group results by category, then by product_id
    categories = defaultdict(lambda: defaultdict(list))
    for result in results:
        if "error" not in result:
            category = result.get("category", "unknown")
            product_id = result.get("product_id", "unknown")
            categories[category][product_id].append(result)
    
    # Compute F1 for each category
    per_category_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    total_categories = 0
    total_products = 0
    total_attributes = 0
    
    for category, products in categories.items():
        category_tp, category_fp, category_fn = 0, 0, 0
        category_products = 0
        category_attributes = 0
        
        # Process each product in this category
        for product_id, product_results in products.items():
            # Build ground truth dict and prediction dict for this product
            gt_dict = {}  # {attr_name: label_value}
            pred_dict = {}  # {attr_name: predicted_value}
            
            try:
                for result in product_results:
                    attr_name = result.get("attr_name", "")
                    label = result.get("label", "")
                    pred = result.get("pred", {})
                    
                    # Extract predicted value from dict
                    if isinstance(pred, dict):
                        pred_value = pred.get(attr_name, "").strip()
                    else:
                        pred_value = str(pred).strip()
                    
                    gt_dict[attr_name] = str(label).strip()
                    pred_dict[attr_name] = pred_value
            except Exception as e:
                print(f"Warning: Skipping product {product_id} in category {category} due to key extraction error: {e}")
                continue
            
            # Compute TP/FP/FN for this product
            product_tp, product_fp, product_fn = 0, 0, 0
            matched_keys = set()
            
            # Check ground truth attributes for Recall
            for attr_name, gt_value in gt_dict.items():
                if attr_name in pred_dict:
                    pred_value = pred_dict[attr_name]
                    
                    if custom_fuzzy_match(gt_value, pred_value, threshold):
                        # Correct match
                        product_tp += 1
                        matched_keys.add(attr_name)
                    else:
                        # Incorrect match
                        product_fn += 1
                    
                else:
                    # Attribute not predicted 漏报 
                    product_fn += 1
            
            # Check predicted attributes not in ground truth for Precise
            for attr_name, pred_value in pred_dict.items():
                if attr_name not in gt_dict:
                    product_fp += 1
                elif attr_name not in matched_keys:
                    # Attribute predicted but not in ground truth 误报 (false positive)
                    product_fp += 1
            
            # Accumulate for category metrics
            category_tp += product_tp
            category_fp += product_fp
            category_fn += product_fn
            category_products += 1
            category_attributes += len(gt_dict)
        
        # Compute metrics for this category
        category_precision = category_tp / (category_tp + category_fp) if (category_tp + category_fp) > 0 else 0.0
        category_recall = category_tp / (category_tp + category_fn) if (category_tp + category_fn) > 0 else 0.0
        category_f1 = 2 * category_precision * category_recall / (category_precision + category_recall) \
                      if (category_precision + category_recall) > 0 else 0.0
        
        per_category_metrics[category] = {
            "precision": round(category_precision, 4),
            "recall": round(category_recall, 4),
            "f1": round(category_f1, 4),
            "tp": category_tp,
            "fp": category_fp,
            "fn": category_fn,
            "product_count": category_products,
            "attr_count": category_attributes,
        }
        
        # Accumulate for overall metrics
        total_tp += category_tp
        total_fp += category_fp
        total_fn += category_fn
        total_categories += 1
        total_products += category_products
        total_attributes += category_attributes
    
    # Compute overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0.0
    
    overall_metrics = {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_categories": total_categories,
        "total_products": total_products,
        "total_attributes": total_attributes,
    }
    
    return per_category_metrics, overall_metrics


class AttributeEvaluator:
    """Evaluates attribute extraction performance"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_sample(self, pred: str, label: str, attr_name: str = "",
                       category: str = "", product_id: str = "", 
                       threshold: float = 0.5) -> Dict:
        """
        Evaluate a single prediction using fuzzy matching.
        
        Args:
            pred: Predicted value
            label: Ground truth value
            attr_name: Attribute name (for logging)
            category: Product category (for logging)
            product_id: Product ID (for logging)
            threshold: Matching threshold (default 0.5)
            
        Returns:
            Result dictionary with metrics
        """
        pred_str = str(pred).strip()
        label_str = str(label).strip()
        
        # Determine match status
        if not pred_str:
            # MLLM failed to extract attribute
            match = False
            tp, fp, fn = 0, 0, 1
        elif not label_str:
            # Predicted something when ground truth is empty
            match = False
            tp, fp, fn = 0, 1, 0
        else:
            # Both non-empty
            match = custom_fuzzy_match(label_str, pred_str, threshold)
            if match:
                tp, fp, fn = 1, 0, 0
            else:
                tp, fp, fn = 0, 0, 1
        
        result = {
            "product_id": product_id,
            "category": category,
            "attr_name": attr_name,
            "pred": pred,
            "label": label,
            "match": match,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        
        self.results.append(result)
        return result
    
    def evaluate_batch(self, predictions: List[str], labels: List[str],
                      attr_names: List[str] = None,
                      categories: List[str] = None,
                      product_ids: List[str] = None) -> List[Dict]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted values
            labels: List of ground truth values
            attr_names: List of attribute names
            categories: List of categories
            product_ids: List of product IDs
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            attr_name = attr_names[i] if attr_names else ""
            category = categories[i] if categories else ""
            product_id = product_ids[i] if product_ids else ""
            
            result = self.evaluate_sample(pred, label, attr_name, category, product_id)
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict:
        """
        Compute aggregate metrics from all evaluated samples.
        
        Returns:
            Dictionary with metrics at different levels (precision, recall, F1)
        """
        if not self.results:
            return {}
        
        # Overall metrics
        total_tp = sum(r["tp"] for r in self.results)
        total_fp = sum(r["fp"] for r in self.results)
        total_fn = sum(r["fn"] for r in self.results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics = {
            "total_samples": len(self.results),
            "overall_precision": round(overall_precision, 4),
            "overall_recall": round(overall_recall, 4),
            "overall_f1": round(overall_f1, 4),
            "overall_accuracy": round(sum(1 for r in self.results if r["match"]) / len(self.results), 4),
            "overall_tp": total_tp,
            "overall_fp": total_fp,
            "overall_fn": total_fn,
        }
        
        # Per-category metrics
        by_category = defaultdict(list)
        for r in self.results:
            by_category[r["category"]].append(r)
        
        metrics["by_category"] = {}
        for category, results in by_category.items():
            cat_tp = sum(r["tp"] for r in results)
            cat_fp = sum(r["fp"] for r in results)
            cat_fn = sum(r["fn"] for r in results)
            
            cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
            cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) \
                     if (cat_precision + cat_recall) > 0 else 0.0
            
            metrics["by_category"][category] = {
                "count": len(results),
                "precision": round(cat_precision, 4),
                "recall": round(cat_recall, 4),
                "f1": round(cat_f1, 4),
                "accuracy": round(sum(1 for r in results if r["match"]) / len(results), 4),
                "tp": cat_tp,
                "fp": cat_fp,
                "fn": cat_fn,
            }
        
        # Per-attribute metrics
        by_attr = defaultdict(list)
        for r in self.results:
            by_attr[r["attr_name"]].append(r)
        
        metrics["by_attribute"] = {}
        for attr_name, results in by_attr.items():
            attr_tp = sum(r["tp"] for r in results)
            attr_fp = sum(r["fp"] for r in results)
            attr_fn = sum(r["fn"] for r in results)
            
            attr_precision = attr_tp / (attr_tp + attr_fp) if (attr_tp + attr_fp) > 0 else 0.0
            attr_recall = attr_tp / (attr_tp + attr_fn) if (attr_tp + attr_fn) > 0 else 0.0
            attr_f1 = 2 * attr_precision * attr_recall / (attr_precision + attr_recall) \
                      if (attr_precision + attr_recall) > 0 else 0.0
            
            metrics["by_attribute"][attr_name] = {
                "count": len(results),
                "precision": round(attr_precision, 4),
                "recall": round(attr_recall, 4),
                "f1": round(attr_f1, 4),
                "accuracy": round(sum(1 for r in results if r["match"]) / len(results), 4),
                "tp": attr_tp,
                "fp": attr_fp,
                "fn": attr_fn,
            }
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print formatted metrics"""
        metrics = self.get_metrics()
        
        if not metrics:
            print("No results to report")
            return
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n🔹 Overall Metrics:")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Precision: {metrics['overall_precision']:.4f}")
        print(f"  Recall: {metrics['overall_recall']:.4f}")
        print(f"  F1 Score: {metrics['overall_f1']:.4f}")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  TP: {metrics['overall_tp']}, FP: {metrics['overall_fp']}, FN: {metrics['overall_fn']}")
        
        if metrics.get("by_category"):
            print(f"\n🔹 Per-Category Metrics (sorted by F1 descending):")
            sorted_categories = sorted(
                metrics["by_category"].items(),
                key=lambda x: x[1]["f1"],
                reverse=True
            )
            for category, cat_metrics in sorted_categories:
                print(f"  {category}:")
                print(f"    Count: {cat_metrics['count']}")
                print(f"    Precision: {cat_metrics['precision']:.4f}, Recall: {cat_metrics['recall']:.4f}, F1: {cat_metrics['f1']:.4f}")
                print(f"    Accuracy: {cat_metrics['accuracy']:.4f}")
                print(f"    TP: {cat_metrics['tp']}, FP: {cat_metrics['fp']}, FN: {cat_metrics['fn']}")
        
        if metrics.get("by_attribute"):
            print(f"\n🔹 Per-Attribute Metrics (sorted by F1 descending):")
            sorted_attrs = sorted(
                metrics["by_attribute"].items(),
                key=lambda x: x[1]["f1"],
                reverse=True
            )
            for attr_name, attr_metrics in sorted_attrs:
                print(f"  {attr_name}:")
                print(f"    Count: {attr_metrics['count']}")
                print(f"    Precision: {attr_metrics['precision']:.4f}, Recall: {attr_metrics['recall']:.4f}, F1: {attr_metrics['f1']:.4f}")
                print(f"    Accuracy: {attr_metrics['accuracy']:.4f}")
                print(f"    TP: {attr_metrics['tp']}, FP: {attr_metrics['fp']}, FN: {attr_metrics['fn']}")
        
        print("\n" + "="*80)
    
    def reset(self) -> None:
        """Reset all results"""
        self.results = []
