
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import pickle
import json

# Загрузка и обучение модели
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение CatBoost
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)

model.fit(X_train_scaled, y_train)

# Сохранение модели и скейлера
model.save_model('heart_disease_model.cbm')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Получение метрик
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Важность признаков
feature_importance = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))

# Сохранение конфигурации
config = {
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'features': X.columns.tolist(),
    'feature_importance': feature_importance
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Модель обучена успешно!")
print(f"  Точность: {accuracy:.2%}")
print(f"  ROC-AUC: {roc_auc:.3f}")
print(f"  Модель сохранена в: heart_disease_model.cbm")
