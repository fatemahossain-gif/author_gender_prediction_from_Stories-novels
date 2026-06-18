"""
Classifier Models for Bengali Author Gender Classification
No pretrained models - using sklearn implementations
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class ClassifierFactory:
    """Factory for creating classifiers"""
    
    @staticmethod
    def create_svm():
        """
        Support Vector Machine classifier
        Good for high-dimensional feature spaces
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
        ])
    
    @staticmethod
    def create_svm_linear():
        """Linear SVM classifier"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, random_state=42))
        ])
    
    @staticmethod
    def create_naive_bayes():
        """
        Gaussian Naive Bayes classifier
        Good baseline for text classification
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ])
    
    @staticmethod
    def create_random_forest():
        """
        Random Forest classifier
        Good for feature importance analysis
        """
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def create_gradient_boosting():
        """
        Gradient Boosting classifier
        Good for high accuracy
        """
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
    
    @staticmethod
    def create_logistic_regression():
        """
        Logistic Regression classifier
        Good baseline with interpretability
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
    
    @staticmethod
    def create_all_classifiers():
        """
        Create all classifier variants
        
        Returns:
            Dict of classifier name -> classifier
        """
        return {
            'SVM_RBF': ClassifierFactory.create_svm(),
            'SVM_Linear': ClassifierFactory.create_svm_linear(),
            'Naive_Bayes': ClassifierFactory.create_naive_bayes(),
            'Random_Forest': ClassifierFactory.create_random_forest(),
            'Gradient_Boosting': ClassifierFactory.create_gradient_boosting(),
            'Logistic_Regression': ClassifierFactory.create_logistic_regression()
        }


class EnsembleClassifier:
    """Ensemble of multiple classifiers"""
    
    def __init__(self, classifiers=None):
        """
        Args:
            classifiers: List of (name, classifier) tuples
        """
        if classifiers is None:
            self.classifiers = list(ClassifierFactory.create_all_classifiers().items())
        else:
            self.classifiers = classifiers
    
    def fit(self, X, y):
        """Train all classifiers"""
        for name, clf in self.classifiers:
            clf.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = []
        for name, clf in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """Predict probability using average probability"""
        probas = []
        for name, clf in self.classifiers:
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X)
                probas.append(proba)
            else:
                # For classifiers without predict_proba, use decision_function
                if hasattr(clf, 'decision_function'):
                    df = clf.decision_function(X)
                    proba = 1 / (1 + np.exp(-df))  # Sigmoid
                    proba = np.column_stack([1 - proba, proba])
                    probas.append(proba)
        
        if probas:
            avg_proba = np.mean(probas, axis=0)
            return avg_proba
        else:
            return None
