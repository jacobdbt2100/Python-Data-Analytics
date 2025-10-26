# Python for Data Analytics — 4-Week Roadmap

### Learning Objectives

- Confident with Python for analytics
- Skilled in Pandas-based wrangling
- Able to generate insights with EDA
- Produce interactive dashboards with Plotly
- Portfolio-ready Python projects

## Key Tools

| Library | Purpose | Notes |
|--------|---------|------|
| Pandas | Data analysis and wrangling | Equivalent to Excel/SQL tables |
| NumPy | Efficient numerical operations | Supports Pandas under the hood |
| Matplotlib | Customisable visualisations | Base layer for plotting in Python |
| Seaborn | Statistical visualisations | Built on Matplotlib; cleaner defaults |
| Plotly | Interactive visuals | Dashboard-friendly |

- Jupyter Notebook (`.ipynb`)

> Workflow: Pandas → Seaborn for quick insights → Matplotlib for fine-tuning → Plotly for dashboards

---

## ✅ Week 1 — Python Basics + Data Structures

**Topics**
- Variables, operators, expressions
- Conditional statements & loops
- Functions, modules
- Lists, Tuples, Sets, Dictionaries

**Example**
```python
def net_price(price, tax=0.05):
    return price + (price * tax)

print(net_price(100))  # 105
```

## ✅ Week 2 — Pandas for Data Wrangling

**Topics**
- Importing CSV/Excel files
- Handling missing values
- Filtering and sorting
- Grouping and summarisation
- Merging and joining datasets

**Example**
```python
import pandas as pd

df = pd.read_csv("sales.csv")
df = df.dropna()
summary = df.groupby("region")["amount"].sum()
print(summary)
```

## ✅ Week 3 — Exploratory Data Analysis (EDA)

**Topics**
- Descriptive statistics
- Feature relationships
- Outlier spotting
- Trend discovery

**Example**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x="region", y="amount")
plt.title("Sales Distribution by Region")
plt.show()
```

## ✅ Week 4 — Data Visualisation (Matplotlib + Seaborn + Plotly)

**Topics**
- Bar, line, scatter, histograms
- Correlations (heatmaps)
- Interactive charts with Plotly

**Examples**
```python
# Seaborn
sns.lineplot(data=df, x="date", y="amount")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend")
plt.show()
```

```python
# Plotly
import plotly.express as px

fig = px.scatter(df, x="amount", y="profit", color="region")
fig.show()
```
