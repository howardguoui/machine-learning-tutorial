import type { TopicContent } from '../types'

export const mlPipelines: TopicContent = {
  id: 'ml-pipelines',
  title: { en: 'Production ML Pipelines', zh: '生产ML流水线' },
  contentType: 'code',
  content: {
    en: `Building a model is only 10% of the work. Production ML requires robust pipelines for training, serving, and monitoring.

## Scikit-learn Pipeline Pattern

\`\`\`python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

# --- Data ---
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'age':    np.random.normal(35, 10, n).astype(int).clip(18, 80),
    'income': np.random.lognormal(10, 1, n),
    'credit': np.random.normal(700, 100, n).clip(300, 850),
    'region': np.random.choice(['north', 'south', 'east', 'west'], n),
    'product':np.random.choice(['A', 'B', 'C'], n),
})
df.loc[np.random.choice(n, 200), 'age'] = np.nan
df.loc[np.random.choice(n, 150), 'income'] = np.nan

y = ((df['credit'] > 650) & (df['income'] > 30000)).astype(int)

# --- Preprocessing ---
num_cols = ['age', 'income', 'credit']
cat_cols = ['region', 'product']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols),
])

# --- Full Pipeline ---
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42)),
])

# --- Train/Test Split ---
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       stratify=y, random_state=42)

# --- Hyperparameter Search ---
param_grid = {
    'classifier__max_depth':     [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__n_estimators':  [100, 200],
}
search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"Best params: {search.best_params_}")
print(f"Best CV AUC: {search.best_score_:.4f}")
print(f"Test AUC:   {search.score(X_test, y_test):.4f}")

# --- Serialize Model ---
joblib.dump(best_model, '/tmp/credit_model.pkl')
print("Model saved to /tmp/credit_model.pkl")

# --- Load and Predict ---
loaded_model = joblib.load('/tmp/credit_model.pkl')
sample = X_test.iloc[:5]
probs = loaded_model.predict_proba(sample)[:, 1]
print(f"Sample predictions: {probs.round(3)}")
\`\`\`

## FastAPI Model Serving

\`\`\`python
# production_api.py — deploy as REST API with FastAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("/tmp/credit_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Customer age")
    income: float = Field(..., ge=0, description="Annual income")
    credit: float = Field(..., ge=300, le=850, description="Credit score")
    region: str = Field(..., description="Geographic region")
    product: str = Field(..., description="Product type")

    @validator('region')
    def validate_region(cls, v):
        valid = {'north', 'south', 'east', 'west'}
        if v.lower() not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v.lower()

class PredictResponse(BaseModel):
    probability: float
    approved: bool
    model_version: str = "1.0.0"

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([request.dict()])
        prob = model.predict_proba(df)[0, 1]
        return PredictResponse(
            probability=round(float(prob), 4),
            approved=prob >= 0.5,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
\`\`\`

## Model Monitoring

\`\`\`python
# Detect data drift and model degradation in production

from scipy.stats import ks_2samp, chi2_contingency
import pandas as pd
import numpy as np

class ModelMonitor:
    """
    Production model monitor: detects data drift and performance degradation.
    """
    def __init__(self, reference_data: pd.DataFrame):
        """
        reference_data: training/baseline data statistics
        """
        self.reference_stats = {}
        for col in reference_data.select_dtypes(include=np.number).columns:
            self.reference_stats[col] = {
                'mean': reference_data[col].mean(),
                'std':  reference_data[col].std(),
                'data': reference_data[col].values,
            }

    def detect_drift(self, current_data: pd.DataFrame, alpha=0.05) -> dict:
        """
        KS test for numerical features.
        Returns dict of {feature: drift_detected}
        """
        results = {}
        for col, stats in self.reference_stats.items():
            if col not in current_data.columns:
                continue
            current_values = current_data[col].dropna().values
            ks_stat, p_value = ks_2samp(stats['data'], current_values)
            drift_detected = p_value < alpha
            results[col] = {
                'ks_statistic': round(ks_stat, 4),
                'p_value':      round(p_value, 4),
                'drift':        drift_detected,
            }
        return results

    def performance_summary(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1':       f1_score(y_true, y_pred),
            'roc_auc':  roc_auc_score(y_true, y_prob),
        }

# Example usage
reference_df = pd.DataFrame({'age': np.random.normal(35, 10, 1000)})
monitor = ModelMonitor(reference_df)

# Simulate production data — shifted distribution (drift!)
production_df = pd.DataFrame({'age': np.random.normal(45, 10, 500)})  # older customers
drift_report = monitor.detect_drift(production_df)
print("Drift Report:")
for feat, result in drift_report.items():
    status = "⚠️ DRIFT" if result['drift'] else "✅ OK"
    print(f"  {feat}: {status} (p={result['p_value']})")
\`\`\`

> **Production ML checklist**: (1) versioned model artifacts, (2) input validation, (3) prediction logging, (4) drift monitoring, (5) A/B testing framework, (6) rollback plan.`,

    zh: `构建模型只占工作的10%。生产ML需要健壮的训练、服务和监控流水线。

## Scikit-learn流水线模式

\`\`\`python
# 完整的生产就绪流水线
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100)),
])

# 超参数搜索
search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
search.fit(X_train, y_train)

# 序列化模型
joblib.dump(best_model, 'credit_model.pkl')
\`\`\`

## FastAPI模型服务

使用FastAPI将模型部署为REST API：
- 输入验证（Pydantic）
- 启动时加载模型
- 错误处理和日志记录
- 健康检查端点

> **生产ML检查清单**：(1) 版本化模型产物，(2) 输入验证，(3) 预测日志，(4) 漂移监控，(5) A/B测试框架，(6) 回滚计划。`,
  },
}

export const hyperparamTuning: TopicContent = {
  id: 'hyperparameter-tuning',
  title: { en: 'Hyperparameter Tuning', zh: '超参数调优' },
  contentType: 'code',
  content: {
    en: `Hyperparameters control the learning process itself — they can't be learned from data and must be set externally.

## Grid Search vs Random Search vs Bayesian Optimization

\`\`\`python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

X, y = make_classification(n_samples=3000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Grid Search: exhaustive ---
param_grid = {
    'n_estimators': [50, 100],
    'max_depth':    [3, 5, 10, None],
    'min_samples_leaf': [1, 5],
}
# Total: 2 × 4 × 2 = 16 combos × 5 folds = 80 fits

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
print(f"Grid Search — Best AUC: {grid.best_score_:.4f}, Params: {grid.best_params_}")

# --- Random Search: efficient for large spaces ---
param_dist = {
    'n_estimators':     randint(50, 500),
    'max_depth':        [3, 5, 10, 15, None],
    'min_samples_leaf': randint(1, 20),
    'max_features':     uniform(0.3, 0.7),
}
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42
)
rand_search.fit(X_train, y_train)
print(f"Random Search — Best AUC: {rand_search.best_score_:.4f}")

# --- Bayesian Optimization with Optuna ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
            'max_depth':        trial.suggest_int('max_depth', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
            'max_features':     trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42,
        }
        clf = RandomForestClassifier(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"Optuna — Best AUC: {study.best_value:.4f}, Params: {study.best_params}")

except ImportError:
    print("Install optuna for Bayesian optimization: pip install optuna")
\`\`\`

## Learning Rate Finder

\`\`\`python
# Find optimal learning rate by running a mini training sweep
def find_learning_rate(model_fn, X_train, y_train,
                        lr_range=(1e-5, 1), n_steps=100):
    """
    Trains with exponentially increasing LR.
    Optimal LR is typically where loss starts increasing.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_steps)
    losses = []

    for lr in lrs:
        clf = SGDClassifier(loss='log_loss', alpha=0, learning_rate='constant',
                           eta0=lr, max_iter=1, warm_start=False, random_state=42)
        clf.fit(X_s, y_train)
        proba = clf.predict_proba(X_s)[:, 1]
        loss = -np.mean(y_train * np.log(proba + 1e-10) +
                       (1-y_train) * np.log(1-proba + 1e-10))
        losses.append((lr, loss))

    # Find steepest negative slope
    smooth_losses = np.array([l for _, l in losses])
    gradients = np.gradient(smooth_losses)
    best_idx = np.argmin(gradients)
    best_lr = losses[best_idx][0]

    print(f"Suggested learning rate: {best_lr:.2e}")
    return best_lr, losses

best_lr, lr_losses = find_learning_rate(None, X_train, y_train)
\`\`\`

| Method | Pros | Cons | Best For |
|---|---|---|---|
| **Grid Search** | Exhaustive, reproducible | Exponential cost | Small search spaces |
| **Random Search** | Fast, works well in practice | May miss regions | Medium spaces |
| **Bayesian (Optuna)** | Smart, efficient | More complex | Large spaces, expensive fits |
| **Population-based** | Parallel, adaptive | Very complex | Neural network training |`,

    zh: `超参数控制学习过程本身——它们无法从数据中学习，必须外部设置。

| 方法 | 优势 | 劣势 | 最适合 |
|---|---|---|---|
| **网格搜索** | 穷举，可重现 | 指数级代价 | 小搜索空间 |
| **随机搜索** | 快速，实践效果好 | 可能错过区域 | 中等空间 |
| **贝叶斯优化(Optuna)** | 智能、高效 | 更复杂 | 大空间，昂贵拟合 |`,
  },
}
