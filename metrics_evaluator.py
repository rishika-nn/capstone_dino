"""
Metrics Evaluator for Video Search System
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    def __init__(self):
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_search_accuracy(self,
                               query_results: List[Dict],
                               ground_truth: List[Dict],
                               timestamp_tolerance: float = 1.0) -> Dict[str, float]:
        """
        Evaluate search result accuracy against ground truth
        
        Args:
            query_results: List of search results from the engine
            ground_truth: List of ground truth annotations
            timestamp_tolerance: Time window (in seconds) to consider results matching
            
        Returns:
            Dictionary with accuracy metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for result in query_results:
            timestamp = result['timestamp']
            matched = False
            
            # Check if result matches any ground truth within tolerance
            for truth in ground_truth:
                if abs(timestamp - truth['timestamp']) <= timestamp_tolerance:
                    true_positives += 1
                    matched = True
                    break
                    
            if not matched:
                false_positives += 1
                
        # Count missed ground truth entries
        for truth in ground_truth:
            matched = False
            for result in query_results:
                if abs(truth['timestamp'] - result['timestamp']) <= timestamp_tolerance:
                    matched = True
                    break
            if not matched:
                false_negatives += 1
        
        # Calculate metrics
        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0
            
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0
            
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
            
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_caption_quality(self,
                               generated_captions: List[str],
                               reference_captions: List[str]) -> Dict[str, float]:
        """
        Evaluate caption quality using BLEU and ROUGE scores
        
        Args:
            generated_captions: List of generated captions
            reference_captions: List of reference/ground truth captions
            
        Returns:
            Dictionary with caption quality metrics
        """
        from nltk.translate.bleu_score import sentence_bleu
        from rouge import Rouge
        
        bleu_scores = []
        rouge = Rouge()
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        
        for gen, ref in zip(generated_captions, reference_captions):
            # BLEU score
            bleu = sentence_bleu([ref.split()], gen.split())
            bleu_scores.append(bleu)
            
            # ROUGE scores
            try:
                scores = rouge.get_scores(gen, ref)[0]
                rouge_scores['rouge-1'].append(scores['rouge-1']['f'])
                rouge_scores['rouge-2'].append(scores['rouge-2']['f'])
                rouge_scores['rouge-l'].append(scores['rouge-l']['f'])
            except:
                continue
                
        return {
            'bleu_score': np.mean(bleu_scores),
            'rouge1_f1': np.mean(rouge_scores['rouge-1']),
            'rouge2_f1': np.mean(rouge_scores['rouge-2']),
            'rougeL_f1': np.mean(rouge_scores['rouge-l'])
        }
    
    def evaluate_temporal_bootstrapping(self,
                                     results_with_bootstrapping: List[Dict],
                                     results_without_bootstrapping: List[Dict],
                                     ground_truth: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the effectiveness of temporal bootstrapping
        
        Args:
            results_with_bootstrapping: Search results with bootstrapping enabled
            results_without_bootstrapping: Search results without bootstrapping
            ground_truth: Ground truth annotations
            
        Returns:
            Dictionary comparing metrics with/without bootstrapping
        """
        metrics_with = self.evaluate_search_accuracy(results_with_bootstrapping, ground_truth)
        metrics_without = self.evaluate_search_accuracy(results_without_bootstrapping, ground_truth)
        
        improvement = {
            'precision_improvement': metrics_with['precision'] - metrics_without['precision'],
            'recall_improvement': metrics_with['recall'] - metrics_without['recall'],
            'f1_improvement': metrics_with['f1_score'] - metrics_without['f1_score']
        }
        
        return {
            'with_bootstrapping': metrics_with,
            'without_bootstrapping': metrics_without,
            'improvements': improvement
        }
    
    def save_evaluation_results(self, results: Dict, experiment_name: str):
        """Save evaluation results to a JSON file"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = self.results_dir / f"{experiment_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {filename}")

    def load_evaluation_results(self, experiment_name: str) -> Optional[Dict]:
        """Load previously saved evaluation results"""
        files = list(self.results_dir.glob(f"{experiment_name}_*.json"))
        if not files:
            return None
            
        latest = max(files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)
            
    def plot_metrics_over_time(self, experiment_name: str):
        """Plot metrics trends over multiple evaluations"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        files = list(self.results_dir.glob(f"{experiment_name}_*.json"))
        if not files:
            logger.warning("No evaluation results found")
            return
            
        data = []
        for f in sorted(files, key=lambda x: x.stat().st_mtime):
            with open(f) as fp:
                result = json.load(fp)
                result['timestamp'] = f.stem.split('_')[-1]
                data.append(result)
                
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['precision'], label='Precision')
        plt.plot(df['timestamp'], df['recall'], label='Recall')
        plt.plot(df['timestamp'], df['f1_score'], label='F1 Score')
        plt.xlabel('Evaluation Timestamp')
        plt.ylabel('Score')
        plt.title(f'Metrics Over Time: {experiment_name}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.results_dir / f"{experiment_name}_trends.png")
        plt.close()

# Example usage:
"""
evaluator = MetricsEvaluator()

# Evaluate search results
metrics = evaluator.evaluate_search_accuracy(
    query_results=search_results,
    ground_truth=ground_truth_annotations
)

# Evaluate caption quality
caption_metrics = evaluator.evaluate_caption_quality(
    generated_captions=generated_captions,
    reference_captions=reference_captions
)

# Save results
evaluator.save_evaluation_results(metrics, "search_accuracy")

# Plot trends
evaluator.plot_metrics_over_time("search_accuracy")
"""