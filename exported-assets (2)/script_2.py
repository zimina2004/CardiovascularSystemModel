# Пересоздаем датасет с более реалистичным распределением заболеваний
np.random.seed(42)

# Генерация синтетических данных для сердечно-сосудистой системы
n_samples = 1000

# Физиологические параметры
age = np.random.normal(55, 15, n_samples)
age = np.clip(age, 20, 90)

# Систолическое и диастолическое давление
systolic_bp = np.random.normal(130, 20, n_samples)
diastolic_bp = np.random.normal(80, 15, n_samples)

# ЧСС
heart_rate = np.random.normal(75, 12, n_samples)

# Холестерин
cholesterol = np.random.normal(200, 40, n_samples)

# ИМТ
bmi = np.random.normal(25, 4, n_samples)

# Глюкоза крови
glucose = np.random.normal(100, 25, n_samples)

# Стресс-индекс (0-10)
stress_level = np.random.uniform(0, 10, n_samples)

# Физическая активность (часов в неделю)
exercise_hours = np.random.exponential(3, n_samples)

# Курение (0-нет, 1-да)
smoking = np.random.binomial(1, 0.3, n_samples)

# Семейная история
family_history = np.random.binomial(1, 0.4, n_samples)

# Пол (0-женский, 1-мужской)
gender = np.random.binomial(1, 0.5, n_samples)

# Создание целевой переменной для различных заболеваний с более реалистичными коэффициентами
# Гипертония (более строгие критерии)
hypertension_risk = (0.003 * age + 
                    0.005 * systolic_bp + 
                    0.003 * diastolic_bp + 
                    0.02 * bmi + 
                    0.15 * smoking + 
                    0.1 * family_history + 
                    0.05 * stress_level - 
                    0.03 * exercise_hours + 
                    np.random.normal(0, 1, n_samples))

hypertension = (hypertension_risk > 1.5).astype(int)

# ИБС (ишемическая болезнь сердца) - более строгие критерии
chd_risk = (0.005 * age + 
           0.002 * systolic_bp + 
           0.001 * cholesterol + 
           0.015 * bmi + 
           0.2 * smoking + 
           0.12 * family_history + 
           0.03 * stress_level - 
           0.02 * exercise_hours + 
           np.random.normal(0, 0.8, n_samples))

coronary_heart_disease = (chd_risk > 1.2).astype(int)

# Диабет - более строгие критерии
diabetes_risk = (0.004 * age + 
                0.001 * systolic_bp + 
                0.003 * glucose + 
                0.025 * bmi + 
                0.08 * family_history + 
                0.02 * stress_level - 
                0.015 * exercise_hours + 
                np.random.normal(0, 0.6, n_samples))

diabetes = (diabetes_risk > 1.0).astype(int)

# Создание датасета
cardiovascular_data = pd.DataFrame({
    'age': age,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'heart_rate': heart_rate,
    'cholesterol': cholesterol,
    'bmi': bmi,
    'glucose': glucose,
    'stress_level': stress_level,
    'exercise_hours': exercise_hours,
    'smoking': smoking,
    'family_history': family_history,
    'gender': gender,
    'hypertension': hypertension,
    'coronary_heart_disease': coronary_heart_disease,
    'diabetes': diabetes
})

# Создание комбинированного показателя сердечно-сосудистого риска
cardiovascular_data['cardiovascular_risk'] = (
    (cardiovascular_data['hypertension'] + 
     cardiovascular_data['coronary_heart_disease'] + 
     cardiovascular_data['diabetes']) >= 1
).astype(int)

print("Датасет с реалистичным распределением создан")
print(f"Размер датасета: {cardiovascular_data.shape}")
print("\nРаспределение заболеваний:")
print(f"Гипертония: {hypertension.sum()} ({hypertension.mean():.2%})")
print(f"ИБС: {coronary_heart_disease.sum()} ({coronary_heart_disease.mean():.2%})")
print(f"Диабет: {diabetes.sum()} ({diabetes.mean():.2%})")
print(f"Общий кардиоваскулярный риск: {cardiovascular_data['cardiovascular_risk'].sum()} ({cardiovascular_data['cardiovascular_risk'].mean():.2%})")

# Сохранение в CSV
cardiovascular_data.to_csv('cardiovascular_disease_dataset.csv', index=False)
print("\nДатасет сохранен в файл 'cardiovascular_disease_dataset.csv'")

# Статистическая сводка
print("\nСтатистическая сводка данных:")
print(cardiovascular_data.describe().round(2))