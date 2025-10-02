# Создаем файл requirements.txt для зависимостей
requirements_content = '''# ЗАВИСИМОСТИ ДЛЯ СИСТЕМЫ АНАЛИЗА СЕРДЕЧНО-СОСУДИСТОГО ЗДОРОВЬЯ
# ================================================================

# Основные библиотеки для машинного обучения
catboost>=1.2.0          # CatBoost - основной ML алгоритм
scikit-learn>=1.3.0      # Инструменты машинного обучения
pandas>=1.5.0            # Работа с данными
numpy>=1.21.0            # Численные вычисления

# Визуализация и графики
matplotlib>=3.5.0        # Создание графиков
seaborn>=0.11.0          # Статистические графики

# Дополнительные инструменты
joblib>=1.2.0            # Сериализация моделей
scipy>=1.9.0             # Научные вычисления

# Для создания exe файла (опционально)
pyinstaller>=5.0.0       # Конвертация в исполняемый файл
'''

# Создаем простую GUI версию с tkinter
gui_code = '''
# ================================================================================
# GUI ПРИЛОЖЕНИЕ ДЛЯ АНАЛИЗА СЕРДЕЧНО-СОСУДИСТОГО ЗДОРОВЬЯ
# Графический интерфейс с CatBoost ML
# ================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import threading
import sys
import os

class HeartHealthGUI:
    """Графический интерфейс для анализа сердечно-сосудистого здоровья"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🫀 Анализ сердечно-сосудистого здоровья")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализация модели
        self.predictor = None
        self.setup_gui()
        
    def setup_gui(self):
        """Настройка графического интерфейса"""
        
        # Заголовок
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(
            title_frame, 
            text="🫀 Система анализа сердечно-сосудистого здоровья",
            font=('Arial', 16, 'bold'),
            fg='white', bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Основной фрейм с вкладками
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Вкладка ввода данных
        self.input_frame = ttk.Frame(notebook)
        notebook.add(self.input_frame, text="📝 Данные пациента")
        self.create_input_tab()
        
        # Вкладка результатов
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="📊 Результаты анализа")
        self.create_results_tab()
        
        # Инициализация модели в фоновом режиме
        self.init_model()
    
    def create_input_tab(self):
        """Создание вкладки для ввода данных пациента"""
        
        # Фрейм для параметров
        params_frame = ttk.LabelFrame(self.input_frame, text="Основные показатели", padding=20)
        params_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Поля ввода
        self.entries = {}
        
        fields = [
            ('age', 'Возраст (лет)', '55'),
            ('systolic_bp', 'Систолическое АД (мм рт.ст.)', '130'),
            ('diastolic_bp', 'Диастолическое АД (мм рт.ст.)', '80'),
            ('heart_rate', 'ЧСС (уд/мин)', '75'),
            ('cholesterol', 'Холестерин (мг/дл)', '200'),
            ('bmi', 'ИМТ (кг/м²)', '25'),
            ('glucose', 'Глюкоза (мг/дл)', '100'),
            ('stress_level', 'Уровень стресса (0-10)', '5'),
            ('exercise_hours', 'Физ. активность (ч/неделю)', '3')
        ]
        
        for i, (key, label, default) in enumerate(fields):
            row = i // 3
            col = i % 3
            
            tk.Label(params_frame, text=label, font=('Arial', 10)).grid(
                row=row*2, column=col, sticky='w', padx=10, pady=5
            )
            
            entry = tk.Entry(params_frame, font=('Arial', 10), width=15)
            entry.insert(0, default)
            entry.grid(row=row*2+1, column=col, padx=10, pady=5)
            self.entries[key] = entry
        
        # Чекбоксы
        checkbox_frame = ttk.LabelFrame(self.input_frame, text="Дополнительные факторы", padding=20)
        checkbox_frame.pack(fill='x', padx=20, pady=10)
        
        self.smoking_var = tk.BooleanVar()
        self.family_history_var = tk.BooleanVar()
        self.gender_var = tk.BooleanVar()
        
        tk.Checkbutton(checkbox_frame, text="Курение", variable=self.smoking_var).pack(side='left', padx=20)
        tk.Checkbutton(checkbox_frame, text="Семейная история", variable=self.family_history_var).pack(side='left', padx=20)
        tk.Checkbutton(checkbox_frame, text="Мужской пол", variable=self.gender_var).pack(side='left', padx=20)
        
        # Кнопка анализа
        analyze_btn = tk.Button(
            self.input_frame,
            text="🔬 Провести анализ",
            font=('Arial', 12, 'bold'),
            bg='#3498db', fg='white',
            command=self.analyze_patient
        )
        analyze_btn.pack(pady=20)
    
    def create_results_tab(self):
        """Создание вкладки для отображения результатов"""
        
        # Текстовое поле для результатов
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame,
            font=('Consolas', 10),
            wrap=tk.WORD,
            width=80,
            height=30
        )
        self.results_text.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Кнопки управления
        buttons_frame = tk.Frame(self.results_frame)
        buttons_frame.pack(fill='x', padx=20, pady=10)
        
        clear_btn = tk.Button(
            buttons_frame,
            text="🗑️ Очистить",
            command=lambda: self.results_text.delete(1.0, tk.END)
        )
        clear_btn.pack(side='left', padx=10)
        
        save_btn = tk.Button(
            buttons_frame,
            text="💾 Сохранить отчет",
            command=self.save_report
        )
        save_btn.pack(side='left', padx=10)
    
    def init_model(self):
        """Инициализация модели машинного обучения"""
        try:
            from cardiovascular_catboost_system import HeartHealthPredictor
            self.predictor = HeartHealthPredictor()
            
            # Создание и обучение модели
            training_data = self.predictor.create_patient_dataset(500)  # Меньше данных для быстроты
            self.predictor.train_disease_prediction_models(training_data)
            
            messagebox.showinfo("Успех", "Модель успешно инициализирована!")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка инициализации модели: {str(e)}")
    
    def get_patient_data(self):
        """Получение данных пациента из формы"""
        try:
            patient_data = {}
            
            # Числовые поля
            for key, entry in self.entries.items():
                patient_data[key] = float(entry.get())
            
            # Булевы поля
            patient_data['smoking'] = int(self.smoking_var.get())
            patient_data['family_history'] = int(self.family_history_var.get())
            patient_data['gender'] = int(self.gender_var.get())
            
            return patient_data
            
        except ValueError as e:
            messagebox.showerror("Ошибка", "Проверьте правильность введенных данных")
            return None
    
    def analyze_patient(self):
        """Анализ данных пациента"""
        if not self.predictor:
            messagebox.showerror("Ошибка", "Модель не инициализирована")
            return
        
        patient_data = self.get_patient_data()
        if not patient_data:
            return
        
        try:
            # Проведение анализа
            analysis = self.predictor.comprehensive_health_analysis(patient_data)
            
            # Создание отчета
            from cardiovascular_catboost_system import MedicalReportGenerator
            report_gen = MedicalReportGenerator()
            report = report_gen.create_patient_report(analysis, patient_data)
            recommendations = report_gen.generate_medical_recommendations(analysis)
            
            # Отображение результатов
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, report)
            self.results_text.insert(tk.END, "\\n\\nРЕКОМЕНДАЦИИ:\\n")
            self.results_text.insert(tk.END, "="*50 + "\\n")
            
            for i, rec in enumerate(recommendations, 1):
                self.results_text.insert(tk.END, f"{i}. {rec}\\n")
            
            # Переключение на вкладку результатов
            notebook = self.root.children['!notebook']
            notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка анализа: {str(e)}")
    
    def save_report(self):
        """Сохранение отчета в файл"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Успех", f"Отчет сохранен: {filename}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")

def main():
    """Запуск GUI приложения"""
    root = tk.Tk()
    app = HeartHealthGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
'''

# Создание bat файла для запуска
bat_file_content = '''@echo off
title Система анализа сердечно-сосудистого здоровья
echo ================================================
echo    ЗАПУСК СИСТЕМЫ АНАЛИЗА СЕРДЦА
echo ================================================
echo.
echo Установка зависимостей...
pip install -r requirements.txt
echo.
echo Запуск приложения...
python cardiovascular_catboost_system.py
echo.
echo Программа завершена. Нажмите любую клавишу...
pause >nul
'''

# Создание инструкции по сборке exe
build_instructions = '''# ИНСТРУКЦИЯ ПО СБОРКЕ .EXE ПРИЛОЖЕНИЯ
# =====================================

## Автоматическая сборка

1. Запустите файл `build_exe.bat` - он автоматически:
   - Установит все зависимости
   - Соберет .exe файл
   - Создаст готовое приложение

## Ручная сборка

1. Установите зависимости:
   ```
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Соберите консольную версию:
   ```
   pyinstaller --onefile cardiovascular_catboost_system.py
   ```

3. Соберите GUI версию:
   ```
   pyinstaller --onefile --windowed cardiovascular_gui.py
   ```

## Файлы проекта

- `cardiovascular_catboost_system.py` - Основная система с CatBoost
- `cardiovascular_gui.py` - Графический интерфейс
- `requirements.txt` - Зависимости проекта
- `run_app.bat` - Запуск через командную строку
- `build_exe.bat` - Автоматическая сборка exe

## Результат

После сборки в папке `dist` появится готовый .exe файл
'''

# Создание bat файла для сборки exe
build_bat_content = '''@echo off
title Сборка EXE приложения
echo ================================================
echo    СБОРКА .EXE ПРИЛОЖЕНИЯ
echo ================================================
echo.
echo Установка PyInstaller...
pip install pyinstaller catboost scikit-learn pandas numpy matplotlib seaborn
echo.
echo Сборка основного приложения...
pyinstaller --onefile --console cardiovascular_catboost_system.py
echo.
echo Сборка GUI приложения...
pyinstaller --onefile --windowed cardiovascular_gui.py
echo.
echo ================================================
echo Готово! Проверьте папку dist/
echo ================================================
pause
'''

# Сохранение всех файлов
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_content)

with open('cardiovascular_gui.py', 'w', encoding='utf-8') as f:
    f.write(gui_code)

with open('run_app.bat', 'w', encoding='utf-8') as f:
    f.write(bat_file_content)

with open('build_exe.bat', 'w', encoding='utf-8') as f:
    f.write(build_bat_content)

with open('BUILD_INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
    f.write(build_instructions)

print("✅ Все файлы созданы успешно:")
print("📄 cardiovascular_catboost_system.py - Основная система")
print("🖥️ cardiovascular_gui.py - Графический интерфейс") 
print("📋 requirements.txt - Зависимости")
print("⚙️ run_app.bat - Запуск приложения")
print("🔨 build_exe.bat - Сборка exe")
print("📖 BUILD_INSTRUCTIONS.md - Инструкции")