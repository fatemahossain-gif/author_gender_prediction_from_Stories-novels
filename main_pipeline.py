"""
Main Pipeline for Bengali Author Gender Classification
Integrates all components: data loading, feature extraction, CV, classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import DataLoader
from feature_extractor import BengaliFeatureExtractor
from cross_validation import LeaveOneAuthorOut, StratifiedAuthorKFold, CrossValidationEvaluator
from classifier_models import ClassifierFactory, EnsembleClassifier

class AuthorGenderClassificationPipeline:
    """Complete classification pipeline"""
    
    def __init__(self, data_dir='.'):
        """
        Args:
            data_dir: Directory containing Female Stories and Male Stories folders
        """
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir)
        self.X = None
        self.y = None
        self.metadata = None
        self.results = {}
    
    def load_data(self):
        """Load all data"""
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        self.X, self.y, self.metadata = self.data_loader.load_data()
        return self.X, self.y, self.metadata
    
    def run_leave_one_author_out(self):
        """
        Run Leave-One-Author-Out cross-validation
        This is the main evaluation strategy to prevent data leakage
        """
        print("\n" + "=" * 60)
        print("STEP 2: Leave-One-Author-Out Cross-Validation")
        print("=" * 60)
        
        if self.X is None:
            self.load_data()
        
        # Create LOAO splitter
        loao = LeaveOneAuthorOut(self.metadata['authors'])
        evaluator = CrossValidationEvaluator(loao)
        
        # Get best classifier (can be tuned)
        clf = ClassifierFactory.create_random_forest()
        
        print(f"\nEvaluating with {loao.n_splits} author-out splits...")
        print(f"Testing on each author individually...\n")
        
        results = evaluator.evaluate(self.X, self.y, clf)
        
        self.results['LOAO'] = {
            'evaluator': evaluator,
            'results': results
        }
        
        print(f"\nLOAO Results:")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"  Std Accuracy: {results['std_accuracy']:.4f}")
        print(f"  Min Accuracy: {results['min_accuracy']:.4f}")
        print(f"  Max Accuracy: {results['max_accuracy']:.4f}")
        
        # Print per-author results
        print(f"\nPer-Author Accuracy:")
        for i, author in enumerate(loao.unique_authors):
            acc = results['fold_accuracies'][i]
            print(f"  {author}: {acc:.4f}")
        
        return results
    
    def run_kfold_validation(self, n_splits=10):
        """
        Run K-Fold cross-validation for comparison
        """
        print("\n" + "=" * 60)
        print(f"STEP 3: {n_splits}-Fold Cross-Validation")
        print("=" * 60)
        
        if self.X is None:
            self.load_data()
        
        # Create stratified K-fold
        kfold = StratifiedAuthorKFold(
            n_splits=n_splits,
            author_list=self.metadata['authors']
        )
        evaluator = CrossValidationEvaluator(kfold)
        
        clf = ClassifierFactory.create_random_forest()
        
        print(f"\nEvaluating with {n_splits}-fold cross-validation...\n")
        
        results = evaluator.evaluate(self.X, self.y, clf)
        
        self.results['KFold'] = {
            'evaluator': evaluator,
            'results': results
        }
        
        print(f"\n{n_splits}-Fold Results:")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"  Std Accuracy: {results['std_accuracy']:.4f}")
        print(f"  Min Accuracy: {results['min_accuracy']:.4f}")
        print(f"  Max Accuracy: {results['max_accuracy']:.4f}")
        
        return results
    
    def compare_classifiers(self, cv_strategy='LOAO'):
        """
        Compare all classifier models
        """
        print("\n" + "=" * 60)
        print(f"STEP 4: Comparing Classifiers ({cv_strategy})")
        print("=" * 60)
        
        if self.X is None:
            self.load_data()
        
        # Choose CV strategy
        if cv_strategy == 'LOAO':
            splitter = LeaveOneAuthorOut(self.metadata['authors'])
        else:
            splitter = StratifiedAuthorKFold(n_splits=10, author_list=self.metadata['authors'])
        
        evaluator = CrossValidationEvaluator(splitter)
        
        # Create all classifiers
        classifiers = ClassifierFactory.create_all_classifiers()
        
        comparison_results = {}
        
        print(f"\nEvaluating {len(classifiers)} classifiers...\n")
        
        for clf_name, clf in classifiers.items():
            print(f"  Testing {clf_name}...", end=' ')
            results = evaluator.evaluate(self.X, self.y, clf)
            comparison_results[clf_name] = results
            print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        self.results['Classifier_Comparison'] = comparison_results
        
        # Print ranking
        print(f"\nClassifier Ranking (by mean accuracy):")
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['mean_accuracy'],
            reverse=True
        )
        
        for rank, (clf_name, results) in enumerate(sorted_results, 1):
            print(f"  {rank}. {clf_name}: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        return comparison_results
    
    def train_final_model(self):
        """
        Train final model on all data for deployment
        (In practice, this would need separate test set evaluation)
        """
        print("\n" + "=" * 60)
        print("STEP 5: Training Final Model")
        print("=" * 60)
        
        if self.X is None:
            self.load_data()
        
        # Use best performer (Random Forest)
        clf = ClassifierFactory.create_random_forest()
        
        print(f"\nTraining Random Forest on all data...")
        clf.fit(self.X, self.y)
        
        self.final_model = clf
        
        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            feature_extractor = BengaliFeatureExtractor()
            feature_names = feature_extractor.get_feature_names()
            
            importances = clf.feature_importances_
            importance_pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\nTop 10 Most Important Features:")
            for i, (fname, importance) in enumerate(importance_pairs[:10], 1):
                print(f"  {i}. {fname}: {importance:.4f}")
        
        return clf
    
    def print_summary(self):
        """
        Print comprehensive summary of all results
        """
        print("\n" + "=" * 60)
        print("SUMMARY OF RESULTS")
        print("=" * 60)
        
        if 'LOAO' in self.results:
            loao_results = self.results['LOAO']['results']
            print(f"\nLeave-One-Author-Out:")
            print(f"  Mean Accuracy: {loao_results['mean_accuracy']:.4f}")
            print(f"  Target Met (>80%): {'YES ✓' if loao_results['mean_accuracy'] > 0.8 else 'NO ✗'}")
        
        if 'KFold' in self.results:
            kfold_results = self.results['KFold']['results']
            print(f"\n10-Fold Cross-Validation:")
            print(f"  Mean Accuracy: {kfold_results['mean_accuracy']:.4f}")
            print(f"  Target Met (>80%): {'YES ✓' if kfold_results['mean_accuracy'] > 0.8 else 'NO ✗'}")
        
        if 'Classifier_Comparison' in self.results:
            print(f"\nBest Classifier:")
            sorted_results = sorted(
                self.results['Classifier_Comparison'].items(),
                key=lambda x: x[1]['mean_accuracy'],
                reverse=True
            )
            best_name, best_results = sorted_results[0]
            print(f"  {best_name}: {best_results['mean_accuracy']:.4f} ± {best_results['std_accuracy']:.4f}")
        
        print(f"\nDataset Info:")
        print(f"  Total Samples: {len(self.y)}")
        print(f"  Female Authors: {sum(1 for g in self.y if g == 0)}")
        print(f"  Male Authors: {sum(1 for g in self.y if g == 1)}")
        print(f"  Unique Authors: {len(set(self.metadata['authors']))}")


def main():
    """
    Main execution
    """
    print("\n" + "#" * 60)
    print("# Bengali Author Gender Classification Pipeline")
    print("# Target: ≥80% Accuracy")
    print("# Strategy: Leave-One-Author-Out Cross-Validation")
    print("#" * 60)
    
    # Create pipeline
    pipeline = AuthorGenderClassificationPipeline(data_dir='.')
    
    # Execute
    pipeline.load_data()
    pipeline.run_leave_one_author_out()
    pipeline.run_kfold_validation(n_splits=10)
    pipeline.compare_classifiers(cv_strategy='LOAO')
    pipeline.train_final_model()
    pipeline.print_summary()
    
    print("\n" + "#" * 60)
    print("# Pipeline Complete!")
    print("#" * 60 + "\n")
    
    return pipeline


if __name__ == '__main__':
    pipeline = main()
