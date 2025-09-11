# Создание синтетических данных для моделирования ССС системы и предсказания заболеваний
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

# Создание целевой переменной для различных заболеваний
# Гипертония
hypertension_risk = (0.01 * age + 
                    0.02 * systolic_bp + 
                    0.015 * diastolic_bp + 
                    0.05 * bmi + 
                    0.3 * smoking + 
                    0.2 * family_history + 
                    0.1 * stress_level - 
                    0.05 * exercise_hours + 
                    np.random.normal(0, 2, n_samples))

hypertension = (hypertension_risk > 5).astype(int)

# ИБС (ишемическая болезнь сердца)
chd_risk = (0.02 * age + 
           0.01 * systolic_bp + 
           0.003 * cholesterol + 
           0.04 * bmi + 
           0.4 * smoking + 
           0.25 * family_history + 
           0.08 * stress_level - 
           0.06 * exercise_hours + 
           np.random.normal(0, 1.5, n_samples))

coronary_heart_disease = (chd_risk > 3).astype(int)

# Диабет
diabetes_risk = (0.015 * age + 
                0.005 * systolic_bp + 
                0.01 * glucose + 
                0.06 * bmi + 
                0.2 * family_history + 
                0.05 * stress_level - 
                0.04 * exercise_hours + 
                np.random.normal(0, 1, n_samples))

diabetes = (diabetes_risk > 2).astype(int)

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

print("Синтетический датасет создан успешно")
print(f"Размер датасета: {cardiovascular_data.shape}")
print("\nРаспределение заболеваний:")
print(f"Гипертония: {hypertension.sum()} ({hypertension.mean():.2%})")
print(f"ИБС: {coronary_heart_disease.sum()} ({coronary_heart_disease.mean():.2%})")
print(f"Диабет: {diabetes.sum()} ({diabetes.mean():.2%})")
print(f"Общий кардиоваскулярный риск: {cardiovascular_data['cardiovascular_risk'].sum()} ({cardiovascular_data['cardiovascular_risk'].mean():.2%})")

# Сохранение первых строк для просмотра
print("\nПервые 5 строк датасета:")
print(cardiovascular_data.head())