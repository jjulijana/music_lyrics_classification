import pandas as pd
import numpy as np
from typing import Dict

class MaxLengthOptimizer:

    @staticmethod
    def analyze_lengths(texts, tokenizer=None, sample_size=10000):
        if len(texts) > sample_size:
            sample_indices = np.random.choice(len(texts), sample_size, replace=False)
            texts_sample = [texts[i] for i in sample_indices]
        else:
            texts_sample = texts
            
        if tokenizer:
            # Transformer
            lengths = []
            for text in texts_sample:
                tokens = tokenizer(text, truncation=False, padding=False)
                lengths.append(len(tokens['input_ids']))
        else:
            # LSTM
            lengths = [len(text.split()) for text in texts_sample]
        
        return {
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "percentile_90": np.percentile(lengths, 90),
            "percentile_95": np.percentile(lengths, 95),
            "percentile_99": np.percentile(lengths, 99),
            "total_samples_analyzed": len(texts_sample)
        }
    
    @staticmethod
    def calculate_optimal_max_length(
        length_stats: Dict,
        coverage_target: float = 0.95,
        efficiency_factor: float = 1.1,
        min_length: int = 32,
        max_length: int = 1024,
        round_to: int = 16
    ) -> int:
        
        if coverage_target >= 0.99:
            target_length = length_stats["percentile_99"]
        elif coverage_target >= 0.95:
            target_length = length_stats["percentile_95"]
        else:
            target_length = length_stats["percentile_90"]
        
        optimal_length = int(target_length * efficiency_factor)
        
        if round_to > 1:
            optimal_length = ((optimal_length + round_to - 1) // round_to) * round_to
        
        optimal_length = max(min_length, min(optimal_length, max_length))
        
        return optimal_length
    
    @staticmethod
    def get_optimal_max_length(
        csv_path: str,
        tokenizer=None,
        coverage_target: float = 0.95,
        efficiency_factor: float = 1.1,
        verbose: bool = False
    ) -> int:
        
        df = pd.read_csv(csv_path)
        texts = df["text"].dropna().tolist()
        
        length_stats = MaxLengthOptimizer.analyze_lengths(texts, tokenizer)
        optimal_length = MaxLengthOptimizer.calculate_optimal_max_length(
            length_stats, coverage_target, efficiency_factor
        )
        
        if verbose:
            print(f"=== Max Length Analysis ===")
            print(f"Dataset: {csv_path}")
            print(f"Samples analyzed: {length_stats['total_samples_analyzed']}")
            print(f"Mean length: {length_stats['mean']:.1f}")
            print(f"95th percentile: {length_stats['percentile_95']:.1f}")
            print(f"Target coverage: {coverage_target*100:.1f}%")
            print(f"Optimal max_length: {optimal_length}")
            
            truncation_rate = (100 - coverage_target * 100)
            print(f"Estimated truncation: ~{truncation_rate:.1f}% of samples")
        
        return optimal_length
