import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

plt.style.use('default')
sns.set_palette("husl")

def plot_overall_metrics(y_true, y_pred):
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics = [test_acc, macro_f1, weighted_f1]
    labels = ['Accuracy', 'Macro F1', 'Weighted F1']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, metrics, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.ylim(0, 1)
    plt.title('Transformer Model - Overall Performance Metrics', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    
    for bar, metric in zip(bars, metrics):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{metric:.3f}', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_f1_per_class(y_true, y_pred, class_names):
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(class_names, f1_per_class, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    plt.ylim(0, 1)
    plt.title('Transformer Model - F1 Score by Emotion Class', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Class', fontsize=14, fontweight='bold')
    
    for bar, f1 in zip(bars, f1_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(y_true, y_prob, class_names):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Transformer Model - ROC Curves per Emotion Class', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curves(y_true, y_prob, class_names):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        avg_prec = average_precision_score(y_bin[:, i], y_prob[:, i])
        
        plt.plot(recall, precision, color=color, linewidth=2,
                label=f'{class_name} (AP = {avg_prec:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Transformer Model - Precision-Recall Curves', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(12, 10))
    mask = cm < 0.01
    
    ax = sns.heatmap(cm, 
                     annot=True, 
                     fmt='.2f', 
                     xticklabels=class_names,
                     yticklabels=class_names, 
                     cmap='Blues',
                     annot_kws={"size": 12, "weight": "bold"},
                     cbar_kws={"shrink": .8},
                     mask=mask)
    
    plt.title('Transformer Model - Normalized Confusion Matrix', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Emotion', fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    acc = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Overall Accuracy: {acc:.1%}', 
             transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    print("=" * 60)
    print(f"TRANSFORMER MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Overall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))