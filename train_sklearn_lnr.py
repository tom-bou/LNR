"""
Sklearn MLP with LNR (Label Noise Rebalancing) for KEEL datasets.

LNR Paper Experimental Setup for KEEL:
- Architecture: MLP with hidden layers (5, 10, 5)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 2000 (based on paper appendix)
- Normalization: Zero mean, unit variance (StandardScaler)
- Evaluation: 5-fold stratified cross-validation
- Metrics: F1 Score, G-Mean, AUC
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from datasets.keel_parser import load_keel_dataset


def g_mean_score(y_true, y_pred):
    """Calculate G-Mean."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    else:
        recalls = np.diag(cm) / cm.sum(axis=1)
        return np.prod(recalls) ** (1/len(recalls))


def apply_lnr(X_train, y_train, model, threshold=3.0):
    """
    Apply Label Noise Rebalancing (LNR) to training labels.
    
    Based on model predictions, flip some majority class labels to minority
    to help balance the decision boundary.
    """
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)
    
    # Get model predictions on training data
    proba = model.predict_proba(X_train)
    
    # For each majority class sample, compute z-score of minority class probability
    majority_mask = y_train == majority_class
    minority_probs = proba[majority_mask, minority_class]
    
    # Compute z-scores
    mean_prob = np.mean(minority_probs)
    std_prob = np.std(minority_probs) + 1e-8
    z_scores = (minority_probs - mean_prob) / std_prob
    
    # Flip labels for samples with high z-score (model thinks they might be minority)
    flip_mask = z_scores > threshold
    
    # Create new labels
    y_new = y_train.copy()
    majority_indices = np.where(majority_mask)[0]
    flip_indices = majority_indices[flip_mask]
    y_new[flip_indices] = minority_class
    
    n_flipped = len(flip_indices)
    print(f"    LNR: Flipped {n_flipped} labels from class {majority_class} to {minority_class}")
    
    return y_new


def train_with_lnr(X_train, y_train, X_test, y_test, hidden_sizes=(5, 10, 5), 
                   max_iter=2000, lnr_threshold=3.0, random_state=42):
    """Train MLP with LNR."""
    
    # Step 1: Train baseline model
    mlp_baseline = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0,
        batch_size=32,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=False,
        verbose=False
    )
    mlp_baseline.fit(X_train, y_train)
    
    # Step 2: Apply LNR to get modified labels
    y_train_lnr = apply_lnr(X_train, y_train, mlp_baseline, threshold=lnr_threshold)
    
    # Step 3: Retrain with modified labels
    mlp_lnr = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0,
        batch_size=32,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state + 1,
        early_stopping=False,
        verbose=False
    )
    mlp_lnr.fit(X_train, y_train_lnr)
    
    # Evaluate both
    results = {}
    
    for name, model in [('Baseline', mlp_baseline), ('LNR', mlp_lnr)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        f1 = f1_score(y_test, y_pred, average='binary')
        g_mean = g_mean_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        
        results[name] = {'f1': f1, 'g_mean': g_mean, 'auc': auc}
    
    return results


def train_cv_with_lnr(dataset_name, hidden_sizes=(5, 10, 5), max_iter=2000, 
                      n_folds=5, lnr_threshold=3.0):
    """Train with 5-fold CV comparing baseline vs LNR."""
    
    # Load data
    dataset_path = Path('./data/keel') / dataset_name
    X_train, y_train, X_test, y_test, _ = load_keel_dataset(dataset_path)
    
    # Combine for CV
    X = np.vstack([X_train, X_test]) if X_test is not None else X_train
    y = np.concatenate([y_train, y_test]) if y_test is not None else y_train
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(X)}, IR: {max(np.bincount(y))/min(np.bincount(y)):.2f}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = {'Baseline': [], 'LNR': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        print(f"  Fold {fold+1}:")
        results = train_with_lnr(X_tr, y_tr, X_te, y_te, hidden_sizes, max_iter, 
                                  lnr_threshold, random_state=42+fold)
        
        for method in ['Baseline', 'LNR']:
            all_results[method].append(results[method])
    
    # Average results
    print(f"\n  Results (5-fold CV):")
    print(f"  {'Method':<12} {'F1':>10} {'G-Mean':>10} {'AUC':>10}")
    print(f"  {'-'*44}")
    
    summary = {}
    for method in ['Baseline', 'LNR']:
        f1s = [r['f1'] for r in all_results[method]]
        gs = [r['g_mean'] for r in all_results[method]]
        aucs = [r['auc'] for r in all_results[method]]
        
        summary[method] = {
            'f1': np.mean(f1s), 'f1_std': np.std(f1s),
            'g_mean': np.mean(gs), 'g_mean_std': np.std(gs),
            'auc': np.mean(aucs), 'auc_std': np.std(aucs)
        }
        
        print(f"  {method:<12} {np.mean(f1s):.3f}±{np.std(f1s):.2f} {np.mean(gs):.3f}±{np.std(gs):.2f} {np.mean(aucs):.3f}±{np.std(aucs):.2f}")
    
    return summary


if __name__ == '__main__':
    datasets = ['yeast1', 'yeast3', 'yeast4', 'yeast5', 'yeast6']
    
    print("="*70)
    print("SKLEARN MLP: Baseline vs LNR (5-Fold CV)")
    print("Architecture: (5, 10, 5), Adam, lr=0.001, 2000 epochs")
    print("LNR threshold: 3.0")
    print("="*70)
    
    all_summaries = {}
    for dataset in datasets:
        all_summaries[dataset] = train_cv_with_lnr(dataset, lnr_threshold=3.0)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY: Baseline vs LNR")
    print("="*80)
    print(f"{'Dataset':<10} | {'Baseline F1':>12} {'G-Mean':>10} {'AUC':>10} | {'LNR F1':>12} {'G-Mean':>10} {'AUC':>10}")
    print("-"*86)
    for dataset, s in all_summaries.items():
        b, l = s['Baseline'], s['LNR']
        print(f"{dataset:<10} | {b['f1']:.3f}±{b['f1_std']:.2f} {b['g_mean']:.3f}±{b['g_mean_std']:.2f} {b['auc']:.3f}±{b['auc_std']:.2f} | {l['f1']:.3f}±{l['f1_std']:.2f} {l['g_mean']:.3f}±{l['g_mean_std']:.2f} {l['auc']:.3f}±{l['auc_std']:.2f}")

