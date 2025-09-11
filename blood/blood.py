import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class CapillaryPressureModel:
    def __init__(self, params=None):
        # нормальные физиологические значения
        default_params = {
            'P_arterial': 35.0,    # Артериальное давление на входе (мм рт.ст.)
            'P_venous': 15.0,      # Венозное давление на выходе (мм рт.ст.)
            'P_interstitial': 5.0, # Интерстициальное давление (мм рт.ст.)
            'L_p': 0.01,           # Гидравлическая проводимость (мл/мин/мм рт.ст./мм²)
            'S': 0.8,             # Коэффициент рефлексии
            'pi_c': 25.0,          # Онкотическое давление крови (мм рт.ст.)
            'pi_i': 10.0,         # Онкотическое давление интерстиция (мм рт.ст.)
            'length': 0.1,         # Длина капилляра (мм)
            'num_points': 100     # Количество точек для расчета
        }
        
        # под замену значений по умолчанию
        self.params = {**default_params, **(params or {})}
        
    def get_user_input(self):
        """Запрашивает параметры у пользователя"""
        print("Введите параметры капиллярного давления (оставьте пустым для значений по умолчанию):")
        
        for param, value in self.params.items():
            if param == 'num_points':
                continue  # Не запрашиваем это у пользователя
            user_input = input(f"{param} (по умолчанию {value}): ")
            if user_input:
                try:
                    self.params[param] = float(user_input)
                except ValueError:
                    print(f"Ошибка! Используется значение по умолчанию {value}")
    
    def pressure_equation(self, P, x):
        """Уравнение для расчета давления вдоль капилляра"""
        return -(self.params['P_arterial'] - self.params['P_venous']) / self.params['length']
    
    def calculate_pressure_profile(self):
        """Рассчитывает профиль давления вдоль капилляра"""
        x = np.linspace(0, self.params['length'], self.params['num_points'])
        P = odeint(self.pressure_equation, self.params['P_arterial'], x)
        return x, P.flatten()
    
    def starling_equation(self, P):
        """Уравнение Старлинга для фильтрации жидкости"""
        return self.params['L_p'] * (
            (P - self.params['P_interstitial']) - 
            self.params['S'] * (self.params['pi_c'] - self.params['pi_i'])
        )
    
    def analyze_edema(self, Q):
        """Анализирует риск развития отека на основе профиля фильтрации"""
        avg_filtration = np.mean(Q)
        net_filtration = np.trapz(Q, dx=self.params['length']/self.params['num_points'])
        
        print("\nАнализ риска отека:")
        print(f"Средняя скорость фильтрации: {avg_filtration:.4f} мл/мин/мм²")
        print(f"Интегральная фильтрация: {net_filtration:.4f} мл/мин")
        
        if net_filtration > 0.5:
            print("Высокий риск отека - значительный избыток фильтрации!")
            edema_mechanism = []
            if self.params['P_arterial'] > 35:
                edema_mechanism.append("повышенное артериальное давление")
            if self.params['P_venous'] > 15:
                edema_mechanism.append("повышенное венозное давление")
            if self.params['pi_c'] < 25:
                edema_mechanism.append("сниженное онкотическое давление крови (гипопротеинемия)")
            if self.params['L_p'] > 0.01:
                edema_mechanism.append("повышенная проницаемость капилляров")
            
            if edema_mechanism:
                print("Возможные механизмы отека: " + ", ".join(edema_mechanism))
        elif net_filtration > 0.2:
            print("Умеренный риск отека")
        else:
            print("Низкий риск отека")
    
    def simulate(self, interactive=False):
        """Выполняет моделирование"""
        if interactive:
            self.get_user_input()
        
        x, P = self.calculate_pressure_profile()
        Q = self.starling_equation(P)
        
        # Визуализация
        plt.figure(figsize=(14, 6))
        
        # График давления
        plt.subplot(1, 3, 1)
        plt.plot(x, P, 'b-', linewidth=2)
        plt.title('Профиль давления в капилляре')
        plt.xlabel('Положение вдоль капилляра (мм)')
        plt.ylabel('Давление (мм рт.ст.)')
        plt.grid(True)
        
        # График фильтрации
        plt.subplot(1, 3, 2)
        plt.plot(x, Q, 'r-', linewidth=2)
        plt.axhline(0, color='k', linestyle='--')
        plt.title('Профиль фильтрации жидкости')
        plt.xlabel('Положение вдоль капилляра (мм)')
        plt.ylabel('Скорость фильтрации (мл/мин/мм²)')
        plt.grid(True)
        
        # График баланса сил
        plt.subplot(1, 3, 3)
        hydrostatic = P - self.params['P_interstitial']
        oncotic = self.params['S'] * (self.params['pi_c'] - self.params['pi_i'])
        plt.plot(x, hydrostatic, 'g-', label='Гидростатическая сила')
        plt.plot(x, oncotic * np.ones_like(x), 'm-', label='Онкотическая сила')
        plt.plot(x, hydrostatic - oncotic, 'k--', label='Результирующая сила')
        plt.legend()
        plt.title('Баланс сил фильтрации')
        plt.xlabel('Положение вдоль капилляра (мм)')
        plt.ylabel('Сила (мм рт.ст.)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Анализ отека
        self.analyze_edema(Q)
        
        return x, P, Q

# Примеры использования
if __name__ == "__main__":
    print("1. Нормальные физиологические условия")
    model_normal = CapillaryPressureModel()
    model_normal.simulate()
    
    print("\n2. Пример отека из-за повышенного венозного давления (сердечная недостаточность)")
    model_heart_failure = CapillaryPressureModel({'P_venous': 25.0})
    model_heart_failure.simulate()
    
    print("\n3. Пример отека из-за низкого онкотического давления (нефротический синдром)")
    model_nephrotic = CapillaryPressureModel({'pi_c': 15.0})
    model_nephrotic.simulate()
    
    print("\n4. Интерактивный режим - введите свои параметры")
    model_interactive = CapillaryPressureModel()
    model_interactive.simulate(interactive=True)