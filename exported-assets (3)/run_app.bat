@echo off
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
