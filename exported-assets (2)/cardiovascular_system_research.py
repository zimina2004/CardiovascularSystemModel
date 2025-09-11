
# ================================================================================
# ИССЛЕДОВАНИЕ ПО МОДЕЛИРОВАНИЮ СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ 
# И ПРЕДСКАЗАНИЮ ЗАБОЛЕВАНИЙ
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# ЦЕЛЬ ПРОЕКТА
# ================================================================================
"""
ЦЕЛЬ ПРОЕКТА:
Разработка комплексной системы моделирования сердечно-сосудистой системы
для предсказания различных заболеваний на основе физиологических параметров
и данных о давлении в сосудах.

ЗАДАЧИ ПРОЕКТА:
1. Создание математической модели гемодинамики сердечно-сосудистой системы
2. Разработка алгоритмов машинного обучения для предсказания:
   - Артериальной гипертензии
   - Ишемической болезни сердца
   - Сахарного диабета
   - Общего кардиоваскулярного риска
3. Создание системы анализа давления в сосудах и кровотока
4. Валидация моделей на клинических данных
5. Разработка интерпретируемой системы принятия решений

ПЛАНИРУЕМЫЙ РЕЗУЛЬТАТ:
- Точность предсказания заболеваний >85%
- Система раннего предупреждения сердечно-сосудистых рисков
- Инструмент для персонализированной медицины
- Платформа для исследований в области кардиологии
"""

# ================================================================================
# КЛАСС МОДЕЛИРОВАНИЯ СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ
# ================================================================================

class CardiovascularSystemModel:
    """
    Комплексная модель для моделирования сердечно-сосудистой системы
    и предсказания заболеваний
    """

    def __init__(self):
        """Инициализация модели"""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'cholesterol', 'bmi', 'glucose', 'stress_level', 
            'exercise_hours', 'smoking', 'family_history', 'gender'
        ]
        self.disease_names = {
            'hypertension': 'Артериальная гипертензия',
            'coronary_heart_disease': 'Ишемическая болезнь сердца',
            'diabetes': 'Сахарный диабет'
        }

    def generate_synthetic_data(self, n_samples=1000):
        """Генерация синтетических данных для обучения"""
        np.random.seed(42)

        # Физиологические параметры
        age = np.random.normal(55, 15, n_samples)
        age = np.clip(age, 20, 90)

        systolic_bp = np.random.normal(130, 20, n_samples)
        diastolic_bp = np.random.normal(80, 15, n_samples)
        heart_rate = np.random.normal(75, 12, n_samples)
        cholesterol = np.random.normal(200, 40, n_samples)
        bmi = np.random.normal(25, 4, n_samples)
        glucose = np.random.normal(100, 25, n_samples)
        stress_level = np.random.uniform(0, 10, n_samples)
        exercise_hours = np.random.exponential(3, n_samples)
        smoking = np.random.binomial(1, 0.3, n_samples)
        family_history = np.random.binomial(1, 0.4, n_samples)
        gender = np.random.binomial(1, 0.5, n_samples)

        # Создание целевых переменных с реалистичными зависимостями
        hypertension_risk = (0.003 * age + 0.005 * systolic_bp + 0.003 * diastolic_bp + 
                            0.02 * bmi + 0.15 * smoking + 0.1 * family_history + 
                            0.05 * stress_level - 0.03 * exercise_hours + 
                            np.random.normal(0, 1, n_samples))
        hypertension = (hypertension_risk > 1.5).astype(int)

        chd_risk = (0.005 * age + 0.002 * systolic_bp + 0.001 * cholesterol + 
                   0.015 * bmi + 0.2 * smoking + 0.12 * family_history + 
                   0.03 * stress_level - 0.02 * exercise_hours + 
                   np.random.normal(0, 0.8, n_samples))
        coronary_heart_disease = (chd_risk > 1.2).astype(int)

        diabetes_risk = (0.004 * age + 0.001 * systolic_bp + 0.003 * glucose + 
                        0.025 * bmi + 0.08 * family_history + 0.02 * stress_level - 
                        0.015 * exercise_hours + np.random.normal(0, 0.6, n_samples))
        diabetes = (diabetes_risk > 1.0).astype(int)

        # Создание DataFrame
        data = pd.DataFrame({
            'age': age, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate, 'cholesterol': cholesterol, 'bmi': bmi,
            'glucose': glucose, 'stress_level': stress_level, 'exercise_hours': exercise_hours,
            'smoking': smoking, 'family_history': family_history, 'gender': gender,
            'hypertension': hypertension, 'coronary_heart_disease': coronary_heart_disease,
            'diabetes': diabetes
        })

        return data

    def train_models(self, data):
        """Обучение моделей машинного обучения"""
        X = data[self.feature_names]
        X_scaled = self.scaler.fit_transform(X)

        results = {}
        diseases = ['hypertension', 'coronary_heart_disease', 'diabetes']

        for disease in diseases:
            print(f"\nОбучение модели для: {self.disease_names[disease]}")

            y = data[disease]
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Тестирование различных алгоритмов
            algorithms = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'Neural Network': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
            }

            best_model = None
            best_score = 0
            algorithm_results = {}

            for name, model in algorithms.items():
                # Кросс-валидация
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                metrics = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_proba)
                }

                algorithm_results[name] = metrics

                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    best_model = model

            self.models[disease] = best_model
            results[disease] = algorithm_results

            print(f"Лучшая модель: {type(best_model).__name__}")
            print(f"CV точность: {best_score:.4f}")

        return results

    def predict_diseases(self, patient_data):
        """Предсказание заболеваний для пациента"""
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = pd.DataFrame([patient_data], columns=self.feature_names)

        X_scaled = self.scaler.transform(patient_df[self.feature_names])

        predictions = {}
        probabilities = {}

        for disease, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]

            predictions[disease] = pred
            probabilities[disease] = prob

        return predictions, probabilities

    def calculate_hemodynamics(self, heart_rate, systolic_bp, diastolic_bp, age, gender):
        """
        Расчет гемодинамических параметров на основе физиологических данных
        """
        # Среднее артериальное давление
        map_pressure = diastolic_bp + (systolic_bp - diastolic_bp) / 3

        # Пульсовое давление
        pulse_pressure = systolic_bp - diastolic_bp

        # Ударный объем (по формуле Старлинга, упрощенная)
        stroke_volume = 70 - (age - 50) * 0.25 + (10 if gender == 1 else 0)
        stroke_volume = max(stroke_volume, 40)

        # Сердечный выброс
        cardiac_output = stroke_volume * heart_rate / 1000  # л/мин

        # Общее периферическое сопротивление
        tpr = (map_pressure * 80) / cardiac_output if cardiac_output > 0 else 0

        # Индекс работы левого желудочка
        lvwi = stroke_volume * map_pressure * 0.0136  # г⋅м/м²

        return {
            'mean_arterial_pressure': round(map_pressure, 2),
            'pulse_pressure': round(pulse_pressure, 2),
            'cardiac_output': round(cardiac_output, 2),
            'total_peripheral_resistance': round(tpr, 2),
            'stroke_volume': round(stroke_volume, 2),
            'left_ventricular_work_index': round(lvwi, 2)
        }

    def simulate_vessel_pressure(self, vessel_radius, vessel_length, cardiac_output, 
                                blood_viscosity=0.004, vessel_compliance=0.1):
        """
        Моделирование давления в сосудах на основе закона Пуазейля 
        и уравнения Windkessel
        """
        # Сопротивление сосуда (закон Пуазейля)
        resistance = (8 * blood_viscosity * vessel_length) / (np.pi * vessel_radius**4)

        # Скорость кровотока
        velocity = cardiac_output / (np.pi * vessel_radius**2 * 60)  # м/с

        # Напряжение сдвига на стенке
        wall_shear_stress = (4 * blood_viscosity * velocity) / vessel_radius

        # Число Рейнольдса
        reynolds_number = (2 * vessel_radius * velocity * 1060) / blood_viscosity

        # Давление в сосуде (модель Windkessel)
        pressure_drop = cardiac_output * resistance * 1000  # Па

        return {
            'vessel_resistance': round(resistance, 6),
            'blood_velocity': round(velocity, 4),
            'wall_shear_stress': round(wall_shear_stress, 4),
            'reynolds_number': round(reynolds_number, 2),
            'pressure_drop': round(pressure_drop, 2)
        }

    def analyze_cardiovascular_risk(self, patient_data):
        """Комплексный анализ сердечно-сосудистого риска"""
        # Предсказание заболеваний
        predictions, probabilities = self.predict_diseases(patient_data)

        # Расчет гемодинамики
        hemodynamics = self.calculate_hemodynamics(
            patient_data['heart_rate'], patient_data['systolic_bp'],
            patient_data['diastolic_bp'], patient_data['age'], patient_data['gender']
        )

        # Расчет общего риска
        total_risk = sum(probabilities.values()) / len(probabilities)

        # Определение уровня риска
        if total_risk < 0.3:
            risk_level = "Низкий"
        elif total_risk < 0.6:
            risk_level = "Умеренный"
        else:
            risk_level = "Высокий"

        return {
            'disease_predictions': predictions,
            'disease_probabilities': probabilities,
            'hemodynamics': hemodynamics,
            'total_cardiovascular_risk': round(total_risk, 4),
            'risk_level': risk_level
        }

# ================================================================================
# СИСТЕМА АНАЛИЗА И ВИЗУАЛИЗАЦИИ
# ================================================================================

class CardiovascularAnalytics:
    """Система аналитики и визуализации результатов"""

    @staticmethod
    def plot_risk_distribution(probabilities):
        """Визуализация распределения рисков заболеваний"""
        diseases = list(probabilities.keys())
        risks = [probabilities[disease] for disease in diseases]

        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = plt.bar(diseases, risks, color=colors, alpha=0.7)

        plt.title('Риск развития сердечно-сосудистых заболеваний', fontsize=14, fontweight='bold')
        plt.ylabel('Вероятность', fontsize=12)
        plt.ylim(0, 1)

        # Добавление значений на столбцы
        for bar, risk in zip(bars, risks):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{risk:.2%}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt

    @staticmethod
    def create_patient_report(analysis_results):
        """Создание отчета по пациенту"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ОТЧЕТ ПО КАРДИОВАСКУЛЯРНОМУ РИСКУ                  ║
╠══════════════════════════════════════════════════════════════════════════════╣

ПРЕДСКАЗАНИЕ ЗАБОЛЕВАНИЙ:
"""

        for disease, prob in analysis_results['disease_probabilities'].items():
            disease_name = {
                'hypertension': 'Артериальная гипертензия',
                'coronary_heart_disease': 'Ишемическая болезнь сердца', 
                'diabetes': 'Сахарный диабет'
            }.get(disease, disease)

            status = "ВЫСОКИЙ РИСК" if prob > 0.6 else "УМЕРЕННЫЙ РИСК" if prob > 0.3 else "НИЗКИЙ РИСК"
            report += f"• {disease_name}: {prob:.1%} ({status})\n"

        report += f"""
ГЕМОДИНАМИЧЕСКИЕ ПАРАМЕТРЫ:
• Среднее артериальное давление: {analysis_results['hemodynamics']['mean_arterial_pressure']} мм рт.ст.
• Пульсовое давление: {analysis_results['hemodynamics']['pulse_pressure']} мм рт.ст.
• Сердечный выброс: {analysis_results['hemodynamics']['cardiac_output']} л/мин
• Ударный объем: {analysis_results['hemodynamics']['stroke_volume']} мл
• Общее периферическое сопротивление: {analysis_results['hemodynamics']['total_peripheral_resistance']} дин⋅с⋅см⁻⁵

ОБЩАЯ ОЦЕНКА РИСКА:
• Интегральный риск: {analysis_results['total_cardiovascular_risk']:.1%}
• Уровень риска: {analysis_results['risk_level']}

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report

# ================================================================================
# ОСНОВНАЯ ПРОГРАММА
# ================================================================================

def main():
    """Основная функция программы"""
    print("═" * 80)
    print("СИСТЕМА МОДЕЛИРОВАНИЯ СЕРДЕЧНО-СОСУДИСТОЙ СИСТЕМЫ")
    print("И ПРЕДСКАЗАНИЯ ЗАБОЛЕВАНИЙ")
    print("═" * 80)

    # Создание модели
    cvs_model = CardiovascularSystemModel()
    analytics = CardiovascularAnalytics()

    # Генерация и загрузка данных
    print("\n1. Генерация синтетических данных...")
    data = cvs_model.generate_synthetic_data(1000)
    print(f"Создан датасет с {len(data)} записями")

    # Обучение моделей
    print("\n2. Обучение моделей машинного обучения...")
    training_results = cvs_model.train_models(data)

    # Отображение результатов обучения
    print("\n" + "─" * 60)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("─" * 60)

    for disease, algorithms in training_results.items():
        disease_name = cvs_model.disease_names[disease]
        print(f"\n{disease_name}:")
        for alg_name, metrics in algorithms.items():
            print(f"  {alg_name:20} | Точность: {metrics['accuracy']:.3f} | AUC: {metrics['auc']:.3f}")

    # Тестирование на примере пациента
    print("\n3. Анализ тестового пациента...")

    test_patient = {
        'age': 55, 'systolic_bp': 150, 'diastolic_bp': 95, 'heart_rate': 80,
        'cholesterol': 240, 'bmi': 28, 'glucose': 110, 'stress_level': 7,
        'exercise_hours': 1, 'smoking': 1, 'family_history': 1, 'gender': 1
    }

    analysis = cvs_model.analyze_cardiovascular_risk(test_patient)
    report = analytics.create_patient_report(analysis)
    print(report)

    # Моделирование давления в сосудах
    print("\n4. Моделирование гемодинамики сосудов...")

    vessel_params = cvs_model.simulate_vessel_pressure(
        vessel_radius=0.002,  # 2 мм
        vessel_length=0.1,    # 10 см
        cardiac_output=analysis['hemodynamics']['cardiac_output']
    )

    print("Параметры кровотока в сосуде:")
    for param, value in vessel_params.items():
        param_names = {
            'vessel_resistance': 'Сопротивление сосуда',
            'blood_velocity': 'Скорость крови (м/с)',
            'wall_shear_stress': 'Напряжение сдвига',
            'reynolds_number': 'Число Рейнольдса',
            'pressure_drop': 'Падение давления (Па)'
        }
        print(f"• {param_names.get(param, param)}: {value}")

    print("\n" + "═" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("═" * 80)

if __name__ == "__main__":
    main()

# ================================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ КЛИНИЧЕСКОГО ПРИМЕНЕНИЯ
# ================================================================================

def clinical_decision_support(patient_data, model):
    """Система поддержки клинических решений"""
    analysis = model.analyze_cardiovascular_risk(patient_data)

    recommendations = []

    # Рекомендации по гипертензии
    if analysis['disease_probabilities']['hypertension'] > 0.6:
        recommendations.append("Рекомендуется консультация кардиолога для оценки артериального давления")
        recommendations.append("Рассмотреть назначение антигипертензивной терапии")

    # Рекомендации по ИБС
    if analysis['disease_probabilities']['coronary_heart_disease'] > 0.6:
        recommendations.append("Рекомендовано проведение ЭКГ и эхокардиографии")
        recommendations.append("Рассмотреть проведение нагрузочных тестов")

    # Рекомендации по диабету
    if analysis['disease_probabilities']['diabetes'] > 0.6:
        recommendations.append("Необходим контроль уровня глюкозы крови")
        recommendations.append("Рекомендуется консультация эндокринолога")

    # Общие рекомендации
    if analysis['total_cardiovascular_risk'] > 0.5:
        recommendations.extend([
            "Рекомендуется увеличение физической активности",
            "Необходима коррекция диеты и образа жизни",
            "Регулярный мониторинг сердечно-сосудистых параметров"
        ])

    return recommendations

def batch_analysis(patients_data, model):
    """Массовый анализ группы пациентов"""
    results = []

    for i, patient in enumerate(patients_data):
        analysis = model.analyze_cardiovascular_risk(patient)
        analysis['patient_id'] = i
        results.append(analysis)

    # Статистика по группе
    high_risk_patients = sum(1 for r in results if r['total_cardiovascular_risk'] > 0.6)
    avg_risk = np.mean([r['total_cardiovascular_risk'] for r in results])

    summary = {
        'total_patients': len(patients_data),
        'high_risk_patients': high_risk_patients,
        'average_risk': avg_risk,
        'detailed_results': results
    }

    return summary

# ================================================================================
# СОХРАНЕНИЕ МОДЕЛИ
# ================================================================================

def save_model(model, filename):
    """Сохранение обученной модели"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в файл: {filename}")

def load_model(filename):
    """Загрузка сохраненной модели"""
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Модель загружена из файла: {filename}")
    return model
