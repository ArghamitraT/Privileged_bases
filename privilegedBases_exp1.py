import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
from scipy.stats import ortho_group
from utility import get_path, create_timestamped_filename

# Set a random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 🔹 Step 1: Dataset with meaningful basis
# ==============================================================================
print("🔹 Step 1: Loading the Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}\n")

# We split the data *before* any transformations to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================================================================
# 🔹 Step 2 & 3: Generate embeddings via metric learning
# ==============================================================================
print("🔹 Step 2 & 3: Generating metric learning embeddings (X') using Isomap...")
# Isomap is a manifold learning technique that seeks to preserve the geodesic
# distances between points. This aligns perfectly with the goal of preserving
# pairwise distances without being tied to the original axes.
# We will embed into the same dimension as the original space for a fair comparison.
n_components = X_train.shape[1]
isomap = Isomap(n_components=n_components, n_neighbors=10)

# Fit on the training data and transform both train and test sets
X_train_prime = isomap.fit_transform(X_train)
X_test_prime = isomap.transform(X_test)
print(f"Generated embeddings with shape: {X_train_prime.shape}\n")


# ==============================================================================
# 🔹 Step 4: Train baseline classifiers
# ==============================================================================
print("🔹 Step 4: Training baseline Decision Trees on original data (X) and embeddings (X')...")

def train_and_evaluate(x_train_data, y_train_data, x_test_data, y_test_data):
    """Helper function to train and evaluate a Decision Tree."""
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(x_train_data, y_train_data)
    preds = clf.predict(x_test_data)
    return accuracy_score(y_test_data, preds)

# Train on original, axis-aligned data
acc_X = train_and_evaluate(X_train, y_train, X_test, y_test)

# Train on learned embeddings
acc_X_prime = train_and_evaluate(X_train_prime, y_train, X_test_prime, y_test)

print(f"Accuracy on original data (X): {acc_X:.4f}")
print(f"Accuracy on embeddings (X'): {acc_X_prime:.4f}\n")


# ==============================================================================
# 🔹 Step 5: Apply random rotations and evaluate
# ==============================================================================
print("🔹 Step 5: Applying random rotations and re-evaluating...")
num_rotations = 500
accuracies_XR = []
accuracies_X_prime_R = []

for i in range(num_rotations):
    # Generate a random orthogonal (rotation) matrix
    R = ortho_group.rvs(dim=n_components)

    # Rotate the original data (XR)
    X_train_R = X_train @ R
    X_test_R = X_test @ R
    acc_xr = train_and_evaluate(X_train_R, y_train, X_test_R, y_test)
    accuracies_XR.append(acc_xr)

    # Rotate the embedded data (X'R)
    X_train_prime_R = X_train_prime @ R
    X_test_prime_R = X_test_prime @ R
    acc_xpr = train_and_evaluate(X_train_prime_R, y_train, X_test_prime_R, y_test)
    accuracies_X_prime_R.append(acc_xpr)

print(f"Completed {num_rotations} rotation experiments.\n")


# ==============================================================================
# 🔹 Step 6: Hypothesis Testing & Visualization
# ==============================================================================
print("🔹 Step 6: Analyzing results and testing the hypothesis...")

avg_acc_XR = np.mean(accuracies_XR)
avg_acc_X_prime_R = np.mean(accuracies_X_prime_R)

drop_X = acc_X - avg_acc_XR
drop_X_prime = acc_X_prime - avg_acc_X_prime_R

print("--- HYPOTHESIS RESULTS ---")
print(f"Original Data (X):")
print(f"  - Before Rotation: {acc_X:.4f}")
print(f"  - Avg. After Rotation (XR): {avg_acc_XR:.4f}")
print(f"  - PERFORMANCE DROP: {drop_X:.4f} ({drop_X/acc_X:.2%})")

print("\nLearned Embeddings (X'):")
print(f"  - Before Rotation: {acc_X_prime:.4f}")
print(f"  - Avg. After Rotation (X'R): {avg_acc_X_prime_R:.4f}")
print(f"  - PERFORMANCE DROP: {drop_X_prime:.4f} ({drop_X_prime/acc_X_prime:.2%})\n")


# --- Visualization ---
# Create a DataFrame for easy plotting with seaborn
results_df = pd.DataFrame({
    'Accuracy': accuracies_XR + accuracies_X_prime_R,
    'DataType': ['Rotated Original (XR)'] * num_rotations + ["Rotated Embedding (X'R)"] * num_rotations
})

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(x='DataType', y='Accuracy', data=results_df, ax=ax, width=0.4)

# Add horizontal lines for the non-rotated baseline accuracies
ax.axhline(y=acc_X, color='r', linestyle='--', label=f'Original (X) Accuracy = {acc_X:.3f}')
ax.axhline(y=acc_X_prime, color='b', linestyle='--', label=f"Embedding (X') Accuracy = {acc_X_prime:.3f}")

ax.set_title('Decision Tree Accuracy: Original Data vs. Embeddings (With and Without Rotation)', fontsize=14)
ax.set_xlabel('Data Type', fontsize=12)
ax.set_ylabel('Model Accuracy', fontsize=12)
ax.legend()
ax.set_ylim(bottom=min(results_df['Accuracy'].min(), acc_X, acc_X_prime) - 0.05, top=1.01)

plt.tight_layout()
pic_path = get_path("code/figure")
plt.savefig(f"{pic_path}/{create_timestamped_filename('privileged_bases_experiment1')}.png", dpi=300)
# plt.show()