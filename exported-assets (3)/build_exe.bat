@echo off
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
