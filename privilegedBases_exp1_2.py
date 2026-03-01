import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
from scipy.stats import ortho_group

np.random.seed(42)

# ----------------------------
# Helpers
# ----------------------------
def load_classification_dataset(name: str):
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
    }
    data = loaders[name]()
    X, y = data.data, data.target
    feature_names = getattr(data, "feature_names", [f"f{i}" for i in range(X.shape[1])])
    return X, y, feature_names

def train_and_evaluate_dt(X_train, y_train, X_test, y_test, max_depth=3):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)

def run_one_dataset(
    X, y,
    test_size=0.3,
    num_rotations=500,
    max_depth=6,
    n_neighbors=10,
):
    # split before any transform
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # standardize using training stats
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # metric-learning-ish embedding (Isomap)
    n_components = X_train_s.shape[1]
    # guard: Isomap neighbors must be < #train
    n_neighbors_eff = min(n_neighbors, max(2, len(X_train_s) - 1))
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors_eff)

    X_train_prime = isomap.fit_transform(X_train_s)
    X_test_prime = isomap.transform(X_test_s)

    # baselines
    acc_X = train_and_evaluate_dt(X_train_s, y_train, X_test_s, y_test, max_depth=max_depth)
    acc_Xp = train_and_evaluate_dt(X_train_prime, y_train, X_test_prime, y_test, max_depth=max_depth)

    # rotations
    XR_accs = []
    XpR_accs = []
    for _ in range(num_rotations):
        R = ortho_group.rvs(dim=n_components)

        X_train_R = X_train_s @ R
        X_test_R = X_test_s @ R
        XR_accs.append(train_and_evaluate_dt(X_train_R, y_train, X_test_R, y_test, max_depth=max_depth))

        X_train_prime_R = X_train_prime @ R
        X_test_prime_R = X_test_prime @ R
        XpR_accs.append(train_and_evaluate_dt(X_train_prime_R, y_train, X_test_prime_R, y_test, max_depth=max_depth))

    XR_accs = np.array(XR_accs)
    XpR_accs = np.array(XpR_accs)

    return {
        "acc_X": acc_X,
        "acc_X_prime": acc_Xp,
        "avg_acc_XR": XR_accs.mean(),
        "avg_acc_XpR": XpR_accs.mean(),
        "drop_X": acc_X - XR_accs.mean(),
        "drop_X_prime": acc_Xp - XpR_accs.mean(),
        "XR_accs": XR_accs,
        "XpR_accs": XpR_accs,
        "n": X.shape[0],
        "d": X.shape[1],
    }

# ----------------------------
# Run many datasets
# ----------------------------
dataset_names = ["iris", "wine", "breast_cancer", "digits"]

all_summaries = []
all_rot_rows = []

for name in dataset_names:
    X, y, feature_names = load_classification_dataset(name)
    out = run_one_dataset(
        X, y,
        num_rotations=500,
        max_depth=3,
        n_neighbors=10,
    )

    all_summaries.append({
        "dataset": name,
        "n": out["n"],
        "d": out["d"],
        "acc_X": out["acc_X"],
        "avg_acc_XR": out["avg_acc_XR"],
        "drop_X": out["drop_X"],
        "acc_X_embed": out["acc_X_prime"],
        "avg_acc_X_embed_R": out["avg_acc_XpR"],
        "drop_X_embed": out["drop_X_prime"],
    })

    # long-format rotation accuracies for plotting
    for a in out["XR_accs"]:
        all_rot_rows.append({"dataset": name, "DataType": "XR", "Accuracy": float(a)})
    for a in out["XpR_accs"]:
        all_rot_rows.append({"dataset": name, "DataType": "X'R", "Accuracy": float(a)})

summary_df = pd.DataFrame(all_summaries).sort_values("drop_X", ascending=False)
rot_df = pd.DataFrame(all_rot_rows)

print(summary_df)
