# Python for Data Analytics — 4-Week Roadmap

## Week 1 — Python Fundamentals & Data Structures (Building a strong foundation in syntax, data types, and logic)

- Variables, input/output, and indentation
- Data types: `int`, `float`, `str`, `bool`
- Lists, tuples, sets, dictionaries
- Control statements (`if`, `for`, `while`)
- Functions and `lambda` expressions

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

## Week 2 — Working with Data, NumPy, and Pandas (Importing, cleaning, and analyzing structured data)

- Importing data from CSV, Excel, and other sources
- Introduction to NumPy arrays
- Creating, slicing, and reshaping arrays
- Pandas DataFrames — creation, selection, filtering
- Handling missing values (`.fillna()`, `.dropna()`)
- Aggregation and grouping
- Basic statistics and correlation

**Sample Codes:**
```python
import numpy as np
import pandas as pd

# Importing data
df_csv = pd.read_csv('sales_data.csv')
df_excel = pd.read_excel('sales_data.xlsx')

# Inspecting data
print(df_csv.head())
print(df_csv.info())

# NumPy example
arr = np.array([10, 20, 30, 40])
print(arr.mean())  # 25.0

# Pandas DataFrame from dictionary
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

Perform exploratory data analysis (EDA) on a Sales dataset — import, clean, and summarise totals, averages, and missing values.

## Week 3 — Data Visualization with Matplotlib & Seaborn (Creating visual insights for presentation and storytelling)

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

Create visuals showing sales trends, top categories, and correlation heatmaps to highlight patterns and insights from a dataset.

## Week 4 — Introduction to Machine Learning with Scikit-learn (Building a simple predictive model)

- Introduction to Scikit-learn
- Train-test split
- Linear Regression model
- Model evaluation (R², MAE, RMSE)
- Saving and loading models with `joblib`

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

> Workflow: Pandas > Seaborn for quick insights > Matplotlib for fine-tuning > Plotly for dashboards

#### `Modules:`
In Python, modules are files that contain Python code — such as functions, classes, and variables — that can be reused in other programs.

#### `Matplotlib vs Seaborn:`

| Aspect            | **Matplotlib**                                                   | **Seaborn**                                                                                            |
| ----------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Purpose**       | A low-level plotting library for detailed control over plots.    | A high-level library built on top of Matplotlib for easier, more attractive statistical visualization. |
| **Ease of Use**   | Requires more code and manual styling.                           | Simpler syntax and automatic styling for better visuals.                                               |
| **Customization** | Extremely flexible—fine control over every plot element.         | Limited flexibility but integrates easily with Matplotlib for custom tweaks.                           |
| **Default Style** | Basic and less aesthetic by default.                             | Comes with appealing themes and colour palettes.                                                       |
| **Best For**      | Creating highly customized plots and complex figure layouts.     | Quick, beautiful plots for statistical data exploration.                                               |
| **Integration**   | Foundation for Seaborn and other libraries like pandas plotting. | Works seamlessly with pandas DataFrames and Matplotlib.                                                |

#### `Pandas Summary Codes:`

1. **`Import / Export Data`**

```python
# Read from a CSV file
pd.read_csv(filename)

# Read from a delimited text file
pd.read_table(filename)

# Read from an Excel file
pd.read_excel(filename)

# Read from a SQL table/database
pd.read_sql(query, connection_object):

# Read from a JSON formatted string, URL, or file
pd.read_json(json_string)

# Parse an HTML URL, string, or file to extract tables to a list of DataFrames
pd.read_html(url)

# Create a DataFrame from a dictionary (keys as column names, values as lists)
pd.DataFrame(dict)

# Write to a CSV file
df.to_csv(filename)

# Write to an Excel file
df.to_excel(filename)

# Write to a SQL table
df.to_sql(table_nm, connection_object)

# Write to a file in JSON format
df.to_json(filename)
```

2. **`Inspect Data`**

```python
# View the first 5 rows of the DataFrame
df.head()

# View the last 5 rows of the DataFrame
df.tail()

# View random 5 rows of the DataFrame
df.sample()

# Get the dimensions of the DataFrame
df.shape

# Get a concise summary of the DataFrame
df.info()

# Summary statistics for numerical columns
df.describe()

# Check data types of columns
df.dtypes

#  List column names
df.columns

# Display the index range
df.index
```

3. **`Select Index Data`**

```python
# Select a single column
df['column']

# Select multiple columns
df[['col1', 'col2']]

# Select the first row by position
df.iloc[0]

# Select the first row by index label
df.loc[0]

# Select a specific element by position
df.iloc[0, 0]

# Select a specific element by label
df.loc[0, 'column']

# Filter rows where column > 5
df[df['col'] > 5]

# Slice rows and columns
df.iloc[0:5, 0:2]

# Set a column as the index
df.set_index('column')
```

4. **`Cleaning Data`**

```python
# Check for null values
df.isnull()

# Check for non-null values
df.notnull()

# Drop rows with null values
df.dropna()

# Replace null values with a specific value
df.fillna(value)

# Replace specific values
df.replace(1, 'one')

# Rename columns
df.rename(columns={'old': 'new'})

# Change data type of a column
df.astype('int')

# Remove duplicate rows
df.drop_duplicates()

# Reset the index
df.reset_index()
```

5. **`Sort & Filter Data`**

```python
df.sort_values('col'): Sort by
column in ascending order.
df.sort_values('col',
ascending=False): Sort by column
in descending order.
df.sort_values(['col1', 'col2'],
ascending=[True, False]): Sort
by multiple columns.
df[df['col'] > 5]: Filter rows
based on condition.
df.query('col > 5'): Filter
using a query string.
df.sample(5): Randomly select 5
rows.
df.nlargest(3, 'col'): Get top 3
rows by column.
df.nsmallest(3, 'col'): Get
bottom 3 rows by column.
df.filter(like='part'): Filter
columns by substring.
```

6. **`Group Data`**

```python
df.groupby('col'): Group by a
column.
df.groupby('col').mean(): Mean
of groups.
df.groupby('col').sum(): Sum of
groups.
df.groupby('col').count(): Count
non-null values in groups.
df.groupby('col')
['other_col'].max(): Max value
in another column for groups.
df.pivot_table(values='col',
index='group', aggfunc='mean'):
Create a pivot table.
df.agg({'col1': 'mean', 'col2':
'sum'}): Aggregate multiple
columns.
df.apply(np.mean): Apply a
function to columns.
df.transform(lambda x: x + 10):
Transform data column-wise.
```

8. **`Merge / Join Data`**

```python
pd.concat([df1, df2]):
Concatenate DataFrames
vertically.
pd.concat([df1, df2], axis=1):
Concatenate DataFrames
horizontally.
df1.merge(df2, on='key'): Merge
two DataFrames on a key.
df1.join(df2): SQL-style join.
df1.append(df2): Append rows of
one DataFrame to another.
pd.merge(df1, df2, how='outer',
on='key'): Outer join.
pd.merge(df1, df2, how='inner',
on='key'): Inner join.
pd.merge(df1, df2, how='left',
on='key'): Left join.
pd.merge(df1, df2, how='right',
on='key'): Right join.
```

9. **`Statistical Operations`**

```python
df.mean(): Column-wise mean.
df.median(): Column-wise
median.
df.std(): Column-wise standard
deviation.
df.var(): Column-wise
variance.
df.sum(): Column-wise sum.
df.min(): Column-wise minimum.
df.max(): Column-wise maximum.
df.count(): Count of non-null
values per column.
df.corr(): Correlation matrix.
```

10. **`Data Visualization`**

```python
df.plot(kind='line'): Line
plot.
df.plot(kind='bar'): Vertical
bar plot.
df.plot(kind='barh'):
Horizontal bar plot.
df.plot(kind='hist'):
Histogram.
df.plot(kind='box'): Box
plot.
df.plot(kind='kde'): Kernel
density estimation plot.
df.plot(kind='pie', y='col'):
Pie chart.
df.plot.scatter(x='c1',
y='c2'): Scatter plot.
df.plot(kind='area'): Area
plot.
```
