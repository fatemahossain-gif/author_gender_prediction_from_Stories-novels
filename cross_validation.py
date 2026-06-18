"""
Cross-Validation Strategy: Leave-One-Author-Out (LOAO)
Prevents data leakage by holding out entire authors for testing
"""

import numpy as np
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict

class LeaveOneAuthorOut:
    """
    Leave-One-Author-Out Cross-Validation
    Ensures no data leakage by leaving out all stories from one author
    """
    
    def __init__(self, author_list: List[str]):
        """
        Args:
            author_list: List of author names (one per story)
        """
        self.author_list = author_list
        self.unique_authors = list(set(author_list))
        self.n_splits = len(self.unique_authors)
    
    def split(self, X, y=None):
        """
        Generate train/test indices for LOAO cross-validation
        
        Yields:
            train_idx: Indices for training (all authors except test author)
            test_idx: Indices for testing (stories from one author)
        """
        for test_author in self.unique_authors:
            test_idx = np.array([i for i, author in enumerate(self.author_list) if author == test_author])
            train_idx = np.array([i for i, author in enumerate(self.author_list) if author != test_author])
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StratifiedAuthorKFold:
    """
    K-Fold Cross-Validation with author stratification
    Ensures each fold has balanced representation of authors and genders
    """
    
    def __init__(self, n_splits: int = 10, author_list: List[str] = None, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds
            author_list: List of author names
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.author_list = author_list
        self.random_state = random_state
        np.random.seed(random_state)
    
    def split(self, X, y=None):
        """
        Generate stratified K-fold splits
        
        Yields:
            train_idx: Training indices
            test_idx: Testing indices
        """
        if self.author_list is None:
            # Fallback to simple K-fold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for train_idx, test_idx in kf.split(X):
                yield train_idx, test_idx
        else:
            # Group by author and create stratified splits
            unique_authors = list(set(self.author_list))
            author_indices = {author: [] for author in unique_authors}
            
            for i, author in enumerate(self.author_list):
                author_indices[author].append(i)
            
            # Shuffle indices
            for author in author_indices:
                np.random.shuffle(author_indices[author])
            
            # Create folds
            fold_indices = [[] for _ in range(self.n_splits)]
            
            for author in unique_authors:
                indices = author_indices[author]
                fold_size = len(indices) // self.n_splits
                
                for fold_idx in range(self.n_splits):
                    start = fold_idx * fold_size
                    if fold_idx == self.n_splits - 1:
                        end = len(indices)
                    else:
                        end = (fold_idx + 1) * fold_size
                    
                    fold_indices[fold_idx].extend(indices[start:end])
            
            # Generate splits
            for test_fold in range(self.n_splits):
                test_idx = np.array(fold_indices[test_fold])
                train_idx = np.array([
                    idx for fold_idx in range(self.n_splits) 
                    if fold_idx != test_fold 
                    for idx in fold_indices[fold_idx]
                ])
                
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class CrossValidationEvaluator:
    """
    Evaluator for cross-validation experiments
    """
    
    def __init__(self, cv_strategy):
        """
        Args:
            cv_strategy: Cross-validation splitter (LOAO or KFold)
        """
        self.cv_strategy = cv_strategy
        self.results = []
    
    def evaluate(self, X, y, classifier):
        """
        Evaluate classifier using cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            classifier: Classifier with fit/predict methods
        
        Returns:
            Dict with results and metrics
        """
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_strategy.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            
            fold_results.append({
                'fold': fold_idx,
                'accuracy': accuracy,
                'n_test_samples': len(test_idx),
                'n_train_samples': len(train_idx),
                'y_pred': y_pred,
                'y_test': y_test
            })
        
        self.results = fold_results
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in fold_results]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'fold_accuracies': accuracies,
            'fold_results': fold_results
        }
