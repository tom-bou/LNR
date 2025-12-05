"""
Sklearn MLP baseline for KEEL datasets - matching the LNR paper methodology.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from datasets.keel_parser import load_keel_dataset


def g_mean_score(y_true, y_pred):
    """Calculate G-Mean (geometric mean of class recalls)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    # Handle binary case
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall of positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall of negative class
        return np.sqrt(sensitivity * specificity)
    else:
        recalls = np.diag(cm) / cm.sum(axis=1)
        return np.prod(recalls) ** (1/len(recalls))


def train_sklearn_mlp(dataset_name, hidden_sizes=(5, 10, 5), max_iter=100, random_state=42):
    """Train sklearn MLP on KEEL dataset."""
    
    # Load data
    dataset_path = Path('./data/keel') / dataset_name
    X_train, y_train, X_test, y_test, class_names = load_keel_dataset(dataset_path)
    
    # Normalize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    
    # Create and train MLP (matching paper: hidden_sizes=(5,10,5), adam, 100 epochs)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0,  # No L2 regularization
        batch_size=32,
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=False,
        verbose=False
    )
    
    mlp.fit(X_train, y_train)
    
    # Predict
    y_pred = mlp.predict(X_test)
    y_proba = mlp.predict_proba(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='binary' if len(class_names) == 2 else 'macro')
    g_mean = g_mean_score(y_test, y_pred)
    
    # AUC
    if len(class_names) == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    print(f"\nResults:")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  G-Mean:   {g_mean:.3f}")
    print(f"  AUC:      {auc:.3f}")
    
    return {'f1': f1, 'g_mean': g_mean, 'auc': auc}


if __name__ == '__main__':
    datasets = ['yeast1', 'yeast3', 'yeast4', 'yeast5', 'yeast6']
    
    print("="*60)
    print("SKLEARN MLP BASELINE (matching LNR paper)")
    print("Architecture: (5, 10, 5), Adam, lr=0.001, 100 epochs")
    print("="*60)
    
    results = {}
    for dataset in datasets:
        results[dataset] = train_sklearn_mlp(dataset)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<10} {'F1':>8} {'G-Mean':>8} {'AUC':>8}")
    print("-"*36)
    for dataset, metrics in results.items():
        print(f"{dataset:<10} {metrics['f1']:>8.3f} {metrics['g_mean']:>8.3f} {metrics['auc']:>8.3f}")


