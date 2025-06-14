import json
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from algorithm import Algorithm

class Experiment:
    """Клас для проведення експериментів з алгоритмами."""
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def experiment_1(self):
        """Експеримент 1: Визначення оптимальних параметрів генетичного алгоритму з можливістю збереження в JSON."""
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        generations_list = [100, 500, 1000]
        stagnant_list = [10, 20, 50]
        results = {}

        for gen in generations_list:
            for stagnant in stagnant_list:
                times, profits = [], []
                for _ in range(5):
                    start_time = time.time()
                    _, profit = Algorithm.genetic(self.data_manager.servers, self.data_manager.requests,
                                                  generations=gen, max_stagnant=stagnant)
                    times.append(time.time() - start_time)
                    profits.append(profit)
                avg_time = sum(times) / 5
                avg_profit = sum(profits) / 5
                balance = avg_profit / avg_time if avg_time > 0 else 0
                results[(gen, stagnant)] = {"time": avg_time, "profit": avg_profit, "balance": balance}

        optimal = max(results.items(), key=lambda x: x[1]["balance"])
        print("\nРезультати Експерименту 1:")
        for (gen, stagnant), data in results.items():
            print(f"Покоління: {gen}, Макс. Застій: {stagnant}, "
                  f"Середній Прибуток: {data['profit']:.2f}, Середній Час: {data['time']:.4f} с, "
                  f"Баланс (прибуток/час): {data['balance']:.2f}")
        print(f"Оптимальна Комбінація: Покоління={optimal[0][0]}, "
              f"Макс. Застій={optimal[0][1]}, "
              f"Баланс (прибуток/час): {optimal[1]['balance']:.2f}")

        save_choice = input("Бажаєте зберегти результати в JSON-файл? (1 - так, 2 - ні): ")
        if save_choice == "1":
            filename = input("Введіть назву JSON-файлу для збереження: ")
            try:
                structured_results = {
                    "results": [
                        {
                            "generations": gen,
                            "max_stagnant": stagnant,
                            "time": data["time"],
                            "profit": data["profit"],
                            "balance": data["balance"]
                        }
                        for (gen, stagnant), data in results.items()
                    ],
                    "optimal": {
                        "generations": optimal[0][0],
                        "max_stagnant": optimal[0][1],
                        "balance": optimal[1]["balance"]
                    }
                }
                with open(filename, 'w') as f:
                    json.dump(structured_results, f, indent=2)
                print(f"Результати успішно збережено в {filename}")
            except Exception as e:
                print(f"Помилка збереження результатів: {e}")

    def experiment_2(self):
        """Експеримент 2: Аналіз впливу варіації прибутку на генетичний алгоритм."""
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        n = len(self.data_manager.requests)
        c = sum(r["c"] for r in self.data_manager.requests) / n
        delta_c = c * 0.1
        profits, times = [], []

        for _ in range(10):
            temp_requests = [{"id": r["id"], "d": r["d"], "c": max(1, r["c"] + random.uniform(-delta_c, delta_c))}
                             for r in self.data_manager.requests]
            start_time = time.time()
            _, profit = Algorithm.genetic(self.data_manager.servers, temp_requests)
            times.append(time.time() - start_time)
            profits.append(profit)

        avg_profit = sum(profits) / 10
        avg_time = sum(times) / 10
        profit_diffs = [p - avg_profit for p in profits]

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), profit_diffs, label="Різниця в критерії", color="green")
        plt.axhline(y=0, color="red", linestyle="--", label="Середній Прибуток")
        plt.xlabel("Номер Завдання")
        plt.ylabel("Різниця Прибутку")
        plt.title("Різниці в Критерії для Завдань")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Середній Прибуток: {avg_profit:.2f}")
        print(f"Середній Час Виконання: {avg_time:.4f} секунд")

    def experiment_3(self):
        """Експеримент 3: Аналіз впливу розміру задачі на точність і час виконання алгоритмів."""
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        n_servers_list = [2, 4, 6, 8, 10]
        n_requests_list = [5, 10, 15, 20, 25]
        greedy_profits, greedy_times = [], []
        genetic_profits, genetic_times = [], []

        for n_s in n_servers_list:
            temp_servers = [{"id": i + 1, "K": random.randint(1, 5), "D": random.randint(100, 500),
                             "used_requests": 0, "used_volume": 0} for i in range(n_s)]
            profits_g, times_g = [], []
            profits_ge, times_ge = [], []
            for n_r in n_requests_list:
                temp_requests = [{"id": j + 1, "d": random.randint(10, 100), "c": random.randint(10, 100)}
                                 for j in range(n_r)]

                start_time = time.time()
                _, profit = Algorithm.greedy(temp_servers, temp_requests)
                times_g.append(time.time() - start_time)
                profits_g.append(profit)

                start_time = time.time()
                _, profit = Algorithm.genetic(temp_servers, temp_requests)
                times_ge.append(time.time() - start_time)
                profits_ge.append(profit)

            greedy_profits.append(sum(profits_g) / len(profits_g))
            greedy_times.append(sum(times_g) / len(times_g))
            genetic_profits.append(sum(profits_ge) / len(profits_ge))
            genetic_times.append(sum(times_ge) / len(times_ge))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(n_servers_list, greedy_profits, label="Жадібний", color="blue")
        plt.plot(n_servers_list, genetic_profits, label="Генетичний", color="orange")
        plt.xlabel("Кількість Серверів")
        plt.ylabel("Середній Прибуток")
        plt.title("Вплив Кількості Серверів на Прибуток")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(n_servers_list, greedy_times, label="Жадібний", color="blue")
        plt.plot(n_servers_list, genetic_times, label="Генетичний", color="orange")
        plt.xlabel("Кількість Серверів")
        plt.ylabel("Середній Час (с)")
        plt.title("Вплив Кількості Серверів на Час Виконання")
        plt.legend()
        plt.grid(True)
        plt.show()