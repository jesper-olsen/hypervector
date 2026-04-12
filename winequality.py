#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


class WineBaseline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        """Loads data, creates binary target, and scales features."""
        self.df = pd.read_csv(self.data_path)

        # 1. Simplify target: 3-5 -> 'bad' (0), 6-8 -> 'good' (1)
        self.df["quality"] = self.df["quality"].apply(lambda x: 0 if x <= 5 else 1)

        X = self.df.drop("quality", axis=1)
        y = self.df["quality"]

        # 2. Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Save test/train split so we can use them externally
        if True:
            # We concatenate X and y so the 'quality' column is included
            train_df = pd.concat([self.X_train, self.y_train], axis=1)
            test_df = pd.concat([self.X_test, self.y_test], axis=1)
            train_df.to_csv("wine_train.csv", index=False)
            test_df.to_csv("wine_test.csv", index=False)
            print("Exported wine_train.csv and wine_test.csv")

        # 3. Scale features (Crucial for LR, SGD, and KNN)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"Dataset loaded. Classes: {np.bincount(y)}")
        print(f"Baseline accuracy (Majority Class): {max(y.mean(), 1-y.mean()):.2f}")

    def plot_correlations(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def evaluate_model(self, model, name):
        """Standardized evaluation for any classifier."""
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, preds)
        # Use decision_function if available (LR, SGD), else predict_proba (RF, KNN)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(self.X_test)[:, 1]
        else:
            probs = model.decision_function(self.X_test)

        auc = roc_auc_score(self.y_test, probs)

        print(f"\n--- {name} ---")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"ROC-AUC:       {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, preds))

        model_bytes = len(pickle.dumps(model))
        print(f"Model size: {model_bytes / 1024:.1f} KB")

        return {"model": name, "acc": acc, "auc": auc}


def main():
    # Initialize and prep data
    # Ensure the path matches your local structure
    path = "DATA/winequality-red.csv"
    baseline = WineBaseline(path)

    try:
        baseline.load_and_preprocess()
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return

    # Define models to test
    model_suite = {
        # Your existing models...
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SGD": SGDClassifier(loss="log_loss", random_state=42),
        # New contenders
        "SVM (RBF)": SVC(C=1.3, kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }

    results = [
        baseline.evaluate_model(model, name) for (name, model) in model_suite.items()
    ]

    # Summary Table
    print("\n" + "=" * 30)
    print("FINAL MODEL COMPARISON")
    print("=" * 30)
    print(pd.DataFrame(results).sort_values(by="acc", ascending=False))


if __name__ == "__main__":
    main()
