"""
Sklearn MLP with 5-fold cross-validation for KEEL datasets - matching LNR paper.
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
    """Calculate G-Mean (geometric mean of class recalls)."""
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


def train_sklearn_mlp_cv(dataset_name, hidden_sizes=(5, 10, 5), max_iter=100, n_folds=5):
    """Train sklearn MLP with 5-fold CV on KEEL dataset."""
    
    # Load ALL data (we'll do our own CV split)
    dataset_path = Path('./data/keel') / dataset_name
    X_train, y_train, X_test, y_test, class_names = load_keel_dataset(dataset_path)
    
    # Combine train and test for CV
    X = np.vstack([X_train, X_test]) if X_test is not None else X_train
    y = np.concatenate([y_train, y_test]) if y_test is not None else y_train
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Imbalance ratio: {max(np.bincount(y)) / min(np.bincount(y)):.2f}")
    
    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    f1_scores = []
    g_mean_scores = []
    auc_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Normalize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Train MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0,
            batch_size=32,
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=max_iter,
            shuffle=True,
            random_state=42 + fold,
            early_stopping=False,
            verbose=False
        )
        
        mlp.fit(X_tr, y_tr)
        
        y_pred = mlp.predict(X_te)
        y_proba = mlp.predict_proba(X_te)
        
        f1 = f1_score(y_te, y_pred, average='binary' if len(np.unique(y)) == 2 else 'macro')
        g_mean = g_mean_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_proba[:, 1]) if len(np.unique(y)) == 2 else roc_auc_score(y_te, y_proba, multi_class='ovr')
        
        f1_scores.append(f1)
        g_mean_scores.append(g_mean)
        auc_scores.append(auc)
    
    print(f"\n5-Fold CV Results:")
    print(f"  F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print(f"  G-Mean:   {np.mean(g_mean_scores):.3f} ± {np.std(g_mean_scores):.3f}")
    print(f"  AUC:      {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    
    return {
        'f1': np.mean(f1_scores), 'f1_std': np.std(f1_scores),
        'g_mean': np.mean(g_mean_scores), 'g_mean_std': np.std(g_mean_scores),
        'auc': np.mean(auc_scores), 'auc_std': np.std(auc_scores)
    }


if __name__ == '__main__':
    datasets = ['yeast1', 'yeast3', 'yeast4', 'yeast5', 'yeast6']
    
    print("="*60)
    print("SKLEARN MLP BASELINE - 5-Fold Cross-Validation")
    print("Architecture: (5, 10, 5), Adam, lr=0.001, 100 epochs")
    print("="*60)
    
    results = {}
    for dataset in datasets:
        results[dataset] = train_sklearn_mlp_cv(dataset)
    
    print("\n" + "="*70)
    print("SUMMARY (5-Fold CV)")
    print("="*70)
    print(f"{'Dataset':<10} {'F1':>12} {'G-Mean':>12} {'AUC':>12}")
    print("-"*48)
    for dataset, m in results.items():
        print(f"{dataset:<10} {m['f1']:.3f}±{m['f1_std']:.2f}  {m['g_mean']:.3f}±{m['g_mean_std']:.2f}  {m['auc']:.3f}±{m['auc_std']:.2f}")


