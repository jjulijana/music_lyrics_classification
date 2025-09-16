from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def build_freq_counter(texts, max_rows=None):
    counter = Counter()
    total_tokens = 0
    for i, text in enumerate(texts):
        if max_rows and i >= max_rows:
            break
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = str(text).split()
        for tok in tokens:
            counter[tok] += 1
            total_tokens += 1
    return counter, total_tokens

def coverage_analysis(counter, total_tokens, thresholds=[0.95, 0.97, 0.99]):
    freqs = np.array([c for _, c in counter.most_common()])
    cum = np.cumsum(freqs) / total_tokens
    
    results = {}
    for t in thresholds:
        k = np.argmax(cum >= t) + 1
        results[f"{int(t*100)}%"] = k
    return results, cum

def plot_coverage(cum):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(cum)+1), cum*100)
    plt.xscale("log")
    plt.xlabel("Top-K tokens (log scale)")
    plt.ylabel("Coverage (%)")
    plt.title("Vocabulary size vs. coverage")
    plt.grid(True, alpha=0.3)
    plt.axhline(95, color="red", linestyle="--", label="95%")
    plt.axhline(99, color="green", linestyle="--", label="99%")
    plt.legend()
    plt.tight_layout()
    plt.show()