import type { TopicContent } from '../types'

export const dataPrep: TopicContent = {
  id: 'data-preparation',
  title: { en: 'Data Preparation & EDA', zh: '数据准备与探索性分析' },
  contentType: 'code',
  content: {
    en: `Data preparation is the foundation of every successful ML project. The classic estimate holds: **80% of ML work is data preparation**.

## Exploratory Data Analysis (EDA)

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# === STEP 1: Understand the data ===
print("Shape:", df.shape)
print("\\nData types:")
print(df.dtypes)
print("\\nDescriptive statistics:")
print(df.describe().round(2))

# === STEP 2: Check for missing values ===
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'count': missing, 'pct': missing_pct})
print("\\nMissing values:")
print(missing_report[missing_report['count'] > 0])

# === STEP 3: Distribution analysis ===
print("\\nClass distribution:")
print(df['target'].value_counts(normalize=True).round(3))

# === STEP 4: Correlation analysis ===
corr = df.drop('target', axis=1).corr()
# Find top correlated pairs
pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
print("\\nTop correlated feature pairs:")
for f1, f2, c in top_pairs:
    print(f"  {f1} × {f2}: {c:.3f}")
\`\`\`

## Data Cleaning

\`\`\`python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

# Simulate messy data
np.random.seed(42)
n = 200
X = pd.DataFrame({
    'age':      np.random.normal(35, 10, n),
    'income':   np.random.lognormal(10, 1, n),
    'score':    np.random.uniform(0, 100, n),
    'category': np.random.choice(['A', 'B', 'C'], n),
})

# Inject missing values and outliers
X.loc[np.random.choice(n, 20), 'age'] = np.nan
X.loc[np.random.choice(n, 15), 'income'] = np.nan
X.loc[np.random.choice(n, 5), 'income'] = 1e8  # outliers

# Strategy 1: Simple imputation
num_cols = ['age', 'income', 'score']
imputer = SimpleImputer(strategy='median')  # robust to outliers
X[num_cols] = imputer.fit_transform(X[num_cols])

# Strategy 2: Handle outliers with capping (winsorization)
def winsorize(series, lower_pct=0.01, upper_pct=0.99):
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)

X['income'] = winsorize(X['income'])

# Verify
print("Missing after imputation:", X.isnull().sum().sum())
print("Income range:", X['income'].min().round(), "—", X['income'].max().round())
\`\`\`

## Feature Engineering

\`\`\`python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Production-ready preprocessing pipeline
num_features = ['age', 'income', 'score']
cat_features = ['category']

# Numerical: impute + scale
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Categorical: impute + one-hot encode
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
])

X_processed = preprocessor.fit_transform(X)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {X_processed.shape}")

# Feature names after transformation
num_out = num_features  # same names after scaling
cat_out = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_features)
all_feature_names = num_out + list(cat_out)
print(f"Features: {all_feature_names}")
\`\`\`

## Train/Validation/Test Split

\`\`\`python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# The golden rule: never touch test data until final evaluation
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=20,
                           n_informative=10, random_state=42)

# Stratified split ensures class balance in each split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Val:   {len(X_val)} ({len(X_val)/len(X):.0%})")
print(f"Test:  {len(X_test)} ({len(X_test)/len(X):.0%})")

# Verify class balance
for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    balance = y_split.mean()
    print(f"{name} positive rate: {balance:.3f}")
\`\`\`

> **Critical rule**: Fit the preprocessor ONLY on training data. Transform validation/test data using the fitted preprocessor. Fitting on test data causes **data leakage** — the model indirectly "sees" test statistics during training, inflating evaluation metrics.`,

    zh: `数据准备是每个成功ML项目的基础。经典估计认为：**ML工作的80%是数据准备**。

## 探索性数据分析（EDA）

\`\`\`python
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# 检查缺失值
missing = df.isnull().sum()
print("缺失值:", missing[missing > 0])

# 类别分布
print("类别分布:", df['target'].value_counts(normalize=True).round(3))
\`\`\`

## 生产就绪的预处理流水线

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 数值特征：填充 + 缩放
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# 类别特征：填充 + 独热编码
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
])
\`\`\`

> **关键规则**：预处理器**仅**在训练数据上拟合。使用已拟合的预处理器转换验证/测试数据。在测试数据上拟合会导致**数据泄露**——模型在训练期间间接"看到"测试统计数据，导致评估指标虚高。`,
  },
}

export const featureEngineering: TopicContent = {
  id: 'feature-engineering',
  title: { en: 'Feature Engineering', zh: '特征工程' },
  contentType: 'code',
  content: {
    en: `Feature engineering is the art of transforming raw data into representations that help models learn effectively.

## Numerical Feature Transformations

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)

# Generate skewed data (common in real-world: income, time, counts)
np.random.seed(42)
income = np.random.lognormal(10, 1.5, 1000)  # heavily right-skewed

print(f"Original - skew: {pd.Series(income).skew():.2f}, range: [{income.min():.0f}, {income.max():.0f}]")

# Log transform: reduces right skew
log_income = np.log1p(income)
print(f"Log     - skew: {pd.Series(log_income).skew():.2f}")

# Box-Cox / Yeo-Johnson: handles zeros and negatives
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
bc_income = pt.fit_transform(income.reshape(-1, 1)).ravel()
print(f"Yeo-Johnson - skew: {pd.Series(bc_income).skew():.2f}")

# When to use each:
# StandardScaler:     data is approx. normal, no outliers
# MinMaxScaler:       need fixed range [0,1], bounded features
# RobustScaler:       outliers present (uses IQR instead of std)
# Log/PowerTransform: data is skewed (income, clicks, durations)
# QuantileTransform:  extreme skew or non-parametric data
\`\`\`

## Categorical Feature Encoding

\`\`\`python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

df = pd.DataFrame({
    'color':   ['red', 'blue', 'green', 'red', 'blue'],
    'size':    ['S', 'M', 'L', 'XL', 'M'],
    'country': ['US', 'CN', 'UK', 'US', 'DE'],
    'target':  [1, 0, 1, 1, 0],
})

# One-Hot Encoding: for nominal categories with few levels
ohe = pd.get_dummies(df['color'], prefix='color')
print("One-hot encoded:\\n", ohe.head())

# Ordinal Encoding: for ordered categories
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_enc'] = oe.fit_transform(df[['size']])
print("\\nOrdinal encoded size:", df['size_enc'].values)

# Target Encoding: for high-cardinality categories (many unique values)
# Maps each category to the mean target value → risk of leakage if not done carefully
# Use k-fold scheme in practice
te = TargetEncoder(random_state=42)
df['country_enc'] = te.fit_transform(df[['country']], df['target'])
print("\\nTarget encoded country:", df['country_enc'].values.round(3))
\`\`\`

## Time-Based Features

\`\`\`python
# Time series feature extraction
timestamps = pd.date_range('2024-01-01', periods=365, freq='D')
df_ts = pd.DataFrame({'date': timestamps, 'value': np.random.randn(365).cumsum()})

df_ts['hour']       = df_ts['date'].dt.hour
df_ts['day_of_week']= df_ts['date'].dt.dayofweek   # 0=Mon, 6=Sun
df_ts['month']      = df_ts['date'].dt.month
df_ts['quarter']    = df_ts['date'].dt.quarter
df_ts['is_weekend'] = df_ts['day_of_week'].isin([5, 6]).astype(int)
df_ts['day_of_year']= df_ts['date'].dt.dayofyear

# Cyclical encoding: encode circular features (hour 23 is close to hour 0)
df_ts['hour_sin'] = np.sin(2 * np.pi * df_ts['day_of_week'] / 7)
df_ts['hour_cos'] = np.cos(2 * np.pi * df_ts['day_of_week'] / 7)
# Now Mon and Sun are numerically close, as they should be

# Lag features for time series
df_ts['value_lag1']  = df_ts['value'].shift(1)   # yesterday
df_ts['value_lag7']  = df_ts['value'].shift(7)   # last week
df_ts['value_roll7'] = df_ts['value'].rolling(7).mean()  # 7-day MA
\`\`\`

## Feature Selection

\`\`\`python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names

# Method 1: Univariate statistical test
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
top_features = feature_names[selector.get_support()]
print("Top 10 features (F-test):", top_features[:5])

# Method 2: Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_names)
print("\\nTop 5 by RF importance:")
print(importances.sort_values(ascending=False).head())

# Method 3: Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=10)
rfe.fit(X, y)
rfe_features = feature_names[rfe.support_]
print("\\nRFE selected:", list(rfe_features[:5]))
\`\`\`

> **Feature engineering is where domain expertise multiplies model performance.** A carefully engineered feature often contributes more than switching to a more complex model.`,

    zh: `特征工程是将原始数据转换为帮助模型有效学习的表示的艺术。

## 数值特征变换

\`\`\`python
import numpy as np
import pandas as pd

# 处理偏斜数据（收入、时间、计数等现实世界中常见）
income = np.random.lognormal(10, 1.5, 1000)

# 对数变换：减少右偏
log_income = np.log1p(income)
print(f"对数变换后偏度: {pd.Series(log_income).skew():.2f}")
\`\`\`

## 分类特征编码

| 编码类型 | 使用场景 |
|---|---|
| 独热编码 | 名义类别，少量级别 |
| 序数编码 | 有序类别（S/M/L/XL） |
| 目标编码 | 高基数类别（许多唯一值） |
| 嵌入 | 极高基数（神经网络） |

> **特征工程是领域专业知识放大模型性能的地方。** 精心设计的特征通常比切换到更复杂的模型贡献更大。`,
  },
}
