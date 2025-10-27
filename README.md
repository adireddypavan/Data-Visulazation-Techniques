# Data-Visulazation-Techniques
python program to Demonstrate Various Visulazation Techniques for given data set


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Map target numbers to species names for better visualization
target_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['target'].map(target_mapping)

# Display first few rows of dataset
print("First 5 rows of the dataset:")
print(df.head())

#Scatter Plot: Sepal Length vs Sepal Width
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.show()

#  Histogram: Distribution of Petal Length
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='petal length (cm)', bins=20, kde=True, color='purple')
plt.title('Distribution of Petal Length')
plt.show()

# Box Plot: Petal Width by Species
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='species', y='petal width (cm)', palette='Set2')
plt.title('Petal Width Distribution by Species')
plt.show()

# Pair Plot: Pairwise Relationships
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df[iris.feature_names].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()
