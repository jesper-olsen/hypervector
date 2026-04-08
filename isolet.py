import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_isolet(path):
    X = []
    y = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            *features, label = parts
            features = list(map(float, features))
            label = int(label.strip().strip("."))  # remove trailing '.'
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_isolet("isolet/isolet1+2+3+4.data")
X_test, y_test = load_isolet("isolet/isolet5.data")

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# Pipeline: scaling + SVM
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale")  # good default
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {acc:.4f}")

model_bytes = len(pickle.dumps(model))
print(f"Model size: {model_bytes / 1024:.1f} KB")

n_sv = sum(model.named_steps['svc'].n_support_)
print(f"Support vectors: {n_sv}/{len(X_train)} ({100*n_sv/len(X_train):.1f}%)")
