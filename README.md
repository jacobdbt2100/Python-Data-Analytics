# Python for Data Analytics — 4-Week Roadmap

## Week 1 — Python Fundamentals & Data Structures

Goal: Build a solid foundation in Python syntax, data types, and logic.
- Variables, input/output, and indentation
- Data types: int, float, str, bool
- Lists, tuples, sets, dictionaries
- Control statements (if, for, while)
- Functions and lambda expressions

**Sample Codes:**
```python
# Basic data types
x = 10
y = 3.5
name = "Jacob"
is_active = True
print(type(x), type(name))  # <class 'int'> <class 'str'>

# Conditional statement
if x > 5:
    print("x is greater than 5")

# Loop example
for i in range(3):
    print("Loop count:", i)

# Function example
def greet(name):
    return f"Hello, {name}!"

print(greet("Jacob"))

# Lambda function
square = lambda n: n**2
print(square(4))  # 16
```

**Mini Project:**

Create a small program to summarise student scores from a CSV file using lists and dictionaries.

## Week 2 — Working with Libraries (NumPy & Pandas)
# ADD DATA IMPORT CSV,EXCEL, ETC

Goal: Learn how to manipulate and analyse structured data.

- Introduction to NumPy arrays
- Creating, slicing, and reshaping arrays
- Pandas DataFrames — creation, selection, filtering
- Handling missing values (.fillna(), .dropna())
- Aggregation and grouping
- Basic statistics and correlation

**Sample Codes:**
```python
import numpy as np
import pandas as pd

# NumPy example
arr = np.array([10, 20, 30, 40])
print(arr.mean())  # 25.0

# Pandas DataFrame
data = {'Name': ['Ayo', 'Tunde', 'Jane'], 'Score': [80, 75, 90]}
df = pd.DataFrame(data)
print(df.head())

# Filtering
print(df[df['Score'] > 80])

# Grouping example
df['Category'] = ['A', 'B', 'A']
print(df.groupby('Category')['Score'].mean())
```

**Mini Project:**

Perform exploratory data analysis (EDA) on a Sales dataset — summarize totals, averages, and missing values.

## Week 3 — Data Visualisation (Matplotlib & Seaborn)

Goal: Learn how to communicate insights visually.

- Introduction to Matplotlib
- Seaborn plots: countplot, scatterplot, heatmap
- Customizing charts (titles, labels, legends)
- Comparing relationships between variables

**Sample Codes:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
data = {'Product': ['A', 'B', 'C'], 'Sales': [200, 150, 300]}
df = pd.DataFrame(data)

# Bar chart
plt.bar(df['Product'], df['Sales'])
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.show()

# Seaborn scatter plot
sns.scatterplot(x='Product', y='Sales', data=df)
plt.show()
```

**Mini Project:**
# why Dashboards in python. Suitable?

Create visual dashboards showing sales trends, top categories, and correlation heatmaps.



## Week 4 — Intro to Machine Learning (Scikit-learn)

Goal: Build a simple predictive model from clean data.

- Introduction to Scikit-learn
- Train-test split
- Linear Regression model
- Model evaluation (R², MAE, RMSE)
- Saving and loading models with joblib ......WHAT's THIS???

**Sample Codes:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    'Hours_Studied': [2, 3, 4, 5, 6],
    'Score': [50, 60, 65, 70, 80]
})

X = df[['Hours_Studied']]
y = df['Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))

# Save model
joblib.dump(model, 'score_predictor.pkl')
```
**Final Project:**

Build a simple model to predict sales or customer churn using Scikit-learn.

### MISCELLANEOUS

#### `Key Tools:`

| Library | Purpose | Notes |
|--------|---------|------|
| Pandas | Data analysis and wrangling | Equivalent to Excel/SQL tables |
| NumPy | Efficient numerical operations | Supports Pandas under the hood |
| Matplotlib | Customizable visualizations | Base layer for plotting in Python |
| Seaborn | Statistical visualizations | Built on Matplotlib; cleaner defaults |
| Plotly | Interactive visuals | Dashboard-friendly |

- Jupyter Notebook (`.ipynb`)

> Workflow: Pandas > Seaborn for quick insights > Matplotlib for fine-tuning > Plotly for dashboards
