# Создаем демонстрационный запуск программы
demo_code = '''
# ================================================================================
# ДЕМОНСТРАЦИЯ РАБОТЫ СИСТЕМЫ МОДЕЛИРОВАНИЯ СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ
# ================================================================================

# Выполнение демонстрации системы
exec(open('cardiovascular_system_research.py').read())

print("\\n" + "="*80)
print("ДЕМОНСТРАЦИЯ ДОПОЛНИТЕЛЬНЫХ ВОЗМОЖНОСТЕЙ СИСТЕМЫ")
print("="*80)

# Создание модели для демонстрации
demo_model = CardiovascularSystemModel()
demo_data = demo_model.generate_synthetic_data(500)  # Уменьшаем для быстроты
demo_model.train_models(demo_data)

# Анализ группы пациентов
print("\\n5. Массовый анализ группы пациентов...")

test_patients = [
    {'age': 45, 'systolic_bp': 130, 'diastolic_bp': 85, 'heart_rate': 70, 
     'cholesterol': 200, 'bmi': 24, 'glucose': 95, 'stress_level': 3, 
     'exercise_hours': 4, 'smoking': 0, 'family_history': 0, 'gender': 0},
    
    {'age': 65, 'systolic_bp': 160, 'diastolic_bp': 100, 'heart_rate': 85, 
     'cholesterol': 280, 'bmi': 30, 'glucose': 130, 'stress_level': 8, 
     'exercise_hours': 1, 'smoking': 1, 'family_history': 1, 'gender': 1},
     
    {'age': 35, 'systolic_bp': 120, 'diastolic_bp': 75, 'heart_rate': 65, 
     'cholesterol': 180, 'bmi': 22, 'glucose': 85, 'stress_level': 2, 
     'exercise_hours': 6, 'smoking': 0, 'family_history': 0, 'gender': 0}
]

group_results = batch_analysis(test_patients, demo_model)

print(f"Общее количество пациентов: {group_results['total_patients']}")
print(f"Пациенты высокого риска: {group_results['high_risk_patients']}")
print(f"Средний риск по группе: {group_results['average_risk']:.1%}")

# Клинические рекомендации для пациента высокого риска
print("\\n6. Система поддержки клинических решений...")
high_risk_patient = test_patients[1]  # Пациент с высокими факторами риска

recommendations = clinical_decision_support(high_risk_patient, demo_model)
print("\\nКлинические рекомендации:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\\n" + "="*80)
print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
print("Система готова к клиническому применению.")
print("="*80)
'''

# Запуск демонстрации
exec(demo_code)