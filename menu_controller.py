from data_manager import DataManager
from experiment import Experiment
from algorithm import Algorithm
from typing import Dict

class MenuController:
    """Клас для керування меню програми."""
    def __init__(self, data_manager: DataManager, experiment: Experiment):
        self.data_manager = data_manager
        self.experiment = experiment

    def display_menu(self):
        """Відображення головного меню та обробка вибору користувача."""
        while True:
            print("""
Меню:
1 - Згенерувати випадкові запити та сервери
2 - Завантажити сервери та запити з файлу
3 - Редагувати дані серверів або запитів
4 - Запустити алгоритми розподілу
5 - Зберегти поточний список серверів і запитів у файл
6 - Вибрати експеримент
7 - Ввести сервери та запити вручну
0 - Вийти з програми
            """)
            option = input("Введіть ваш вибір: ")

            if option == "1":
                self.data_manager.generate_data()
            elif option == "2":
                self.data_manager.load_data()
            elif option == "3":
                self.data_manager.edit_data()
            elif option == "4":
                if not self.data_manager.servers or not self.data_manager.requests:
                    print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
                    continue
                algo_choice = input("\nВиберіть алгоритм:\n1 - Жадібний алгоритм\n2 - Генетичний алгоритм\nВаш вибір: ")
                if algo_choice == "1":
                    result, profit = Algorithm.greedy(self.data_manager.servers, self.data_manager.requests)
                    self._display_results(result, profit, "Жадібний алгоритм")
                elif algo_choice == "2":
                    result, profit = Algorithm.genetic(self.data_manager.servers, self.data_manager.requests)
                    self._display_results(result, profit, "Генетичний алгоритм")
                else:
                    print("Неправильний вибір алгоритму.")
            elif option == "5":
                self.data_manager.save_data()
            elif option == "6":
                self._display_experiment_menu()
            elif option == "7":
                self.data_manager.manual_input()
            elif option == "0":
                print("Вихід із програми...")
                break
            else:
                print("Неправильна опція, спробуйте ще раз.")

    def _display_experiment_menu(self):
        """Відображення підменю експериментів."""
        choice = input("\nПідменю Експериментів:\n"
                       "1 - Визначити параметри завершення генетичного алгоритму\n"
                       "2 - Аналіз впливу параметрів генетичного алгоритму на ефективність\n"
                       "3 - Визначити вплив параметрів задачі на точність і час алгоритмів\n"
                       "Виберіть номер експерименту: ")
        if choice == "1":
            self.experiment.experiment_1()
        elif choice == "2":
            self.experiment.experiment_2()
        elif choice == "3":
            self.experiment.experiment_3()
        else:
            print("Неправильний вибір експерименту.")

    def _display_results(self, result: Dict, profit: float, algo_name: str):
        """Відображення результатів алгоритму."""
        print(f"\nРезультати {algo_name}:")
        print(f"Загальний Прибуток: {profit}")
        for i, s in result.items():
            server = self.data_manager.servers[i]
            print(f"Сервер {i + 1} (K={server['K']}, D={server['D']}): "
                  f"Використані запити {s['n']}/{server['K']}, Обсяг {s['used_volume']}/{server['D']}, "
                  f"Запити {list(s['R'])}")