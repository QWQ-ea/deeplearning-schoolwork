import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels
# Initialize the decision tree classifier
clf = DecisionTreeClassifier()
# Fit the classifier on the data
clf.fit(X, y)
# Output the decision tree construction process
print("Decision Tree Construction Process:")
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
# Analyze each feature selection for classification
print("\nFeature Importance Analysis:")
for feature_name, importance in zip(data.feature_names, clf.feature_importances_):
    print(f"Selected Feature: {feature_name} | Importance: {importance}")
