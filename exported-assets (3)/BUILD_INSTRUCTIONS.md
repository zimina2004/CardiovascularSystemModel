# ИНСТРУКЦИЯ ПО СБОРКЕ .EXE ПРИЛОЖЕНИЯ
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
