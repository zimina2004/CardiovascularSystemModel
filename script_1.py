
# Используем RandomForest вместо CatBoost для демонстрации
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Обучение RandomForest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Сохранение модели и скейлера
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
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
print(f"  Признаки: {len(X.columns)}")
print(f"\nТоп-5 важных признаков:")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
for feat, imp in sorted_features:
    print(f"  - {feat}: {imp:.4f}")
