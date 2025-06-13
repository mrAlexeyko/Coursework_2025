import json
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time


class DataManager:
    def __init__(self):
        self.servers = []
        self.requests = []

    def generate_data(self):
        n_servers = self._get_positive_int("Введіть кількість серверів: ")
        n_requests = self._get_positive_int("Введіть кількість запитів: ")

        min_K = self._get_positive_int("Введіть мінімальне значення K: ")
        max_K = self._get_positive_int(f"Введіть максимальне значення K (>{min_K}): ")
        while max_K <= min_K:
            print(f"Максимальне значення має бути більше {min_K}.")
            max_K = self._get_positive_int(f"Введіть максимальне значення K (>{min_K}): ")

        min_D = self._get_positive_int("Введіть мінімальне значення D: ")
        max_D = self._get_positive_int(f"Введіть максимальне значення D (>{min_D}): ")
        while max_D <= min_D:
            print(f"Максимальне значення має бути більше {min_D}.")
            max_D = self._get_positive_int(f"Введіть максимальне значення D (>{min_D}): ")

        min_d = self._get_positive_int("Введіть мінімальне значення d: ")
        max_d = self._get_positive_int(f"Введіть максимальне значення d (>{min_d}): ")
        while max_d <= min_d:
            print(f"Максимальне значення має бути більше {min_d}.")
            max_d = self._get_positive_int(f"Введіть максимальне значення d (>{min_d}): ")

        min_c = self._get_positive_int("Введіть мінімальне значення c: ")
        max_c = self._get_positive_int(f"Введіть максимальне значення c (>{min_c}): ")
        while max_c <= min_c:
            print(f"Максимальне значення має бути більше {min_c}.")
            max_c = self._get_positive_int(f"Введіть максимальне значення c (>{min_c}): ")

        self.servers = [
            {"id": i + 1, "K": random.randint(min_K, max_K), "D": random.randint(min_D, max_D), "used_requests": 0,
             "used_volume": 0}
            for i in range(n_servers)
        ]
        self.requests = [
            {"id": j + 1, "d": random.randint(min_d, max_d), "c": random.randint(min_c, max_c)}
            for j in range(n_requests)
        ]

        print("\nЗгенеровані сервери:")
        for s in self.servers:
            print(s)

        print("\nЗгенеровані запити:")
        for r in self.requests:
            print(r)

    def load_data(self):
        filename = input("Введіть назву JSON-файлу: ")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.servers = data.get("servers", [])
                self.requests = data.get("requests", [])
                for s in self.servers:
                    s.setdefault("used_requests", 0)
                    s.setdefault("used_volume", 0)
            print("\nДані успішно завантажено.")
        except Exception as e:
            print(f"Помилка завантаження: {e}")

    def save_data(self):
        if not self.servers or not self.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані.")
            return
        filename = input("Введіть ім’я JSON-файлу для збереження: ")
        try:
            with open(filename, 'w') as f:
                json.dump({"servers": self.servers, "requests": self.requests}, f, indent=2)
            print("Дані успішно збережено.")
        except Exception as e:
            print(f"Помилка збереження: {e}")

    def edit_data(self):
        if not self.servers and not self.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані.")
            return
        choice = input("Редагувати (1 – сервери, 2 – запити): ")
        if choice == "1" and self.servers:
            for i, s in enumerate(self.servers):
                print(f"{i + 1}. {s}")
            idx = self._get_valid_index(len(self.servers), "Введіть номер сервера для редагування: ") - 1
            self.servers[idx]["K"] = self._get_positive_int("Нове значення K: ")
            self.servers[idx]["D"] = self._get_positive_int("Нове значення D: ")
        elif choice == "2" and self.requests:
            for i, r in enumerate(self.requests):
                print(f"{i + 1}. {r}")
            idx = self._get_valid_index(len(self.requests), "Введіть номер запиту для редагування: ") - 1
            self.requests[idx]["d"] = self._get_positive_int("Нове значення d: ")
            self.requests[idx]["c"] = self._get_positive_int("Нове значення c: ")
        else:
            print("Невірний вибір або відсутні дані.")

    def manual_input(self):
        self.servers = []
        self.requests = []
        n = self._get_positive_int("Введіть кількість серверів: ")
        for i in range(n):
            K = self._get_positive_int(f"Сервер {i + 1} – K: ")
            D = self._get_positive_int(f"Сервер {i + 1} – D: ")
            self.servers.append({"id": i + 1, "K": K, "D": D, "used_requests": 0, "used_volume": 0})

        m = self._get_positive_int("Введіть кількість запитів: ")
        for j in range(m):
            d = self._get_positive_int(f"Запит {j + 1} – d: ")
            c = self._get_positive_int(f"Запит {j + 1} – c: ")
            self.requests.append({"id": j + 1, "d": d, "c": c})

        print("\nВведені сервери:")
        for s in self.servers:
            print(s)

        print("\nВведені запити:")
        for r in self.requests:
            print(r)

    def _get_positive_int(self, prompt):
        while True:
            try:
                value = int(input(prompt))
                if value <= 0:
                    print("Введіть додатне число.")
                    continue
                return value
            except ValueError:
                print("Введіть коректне ціле число.")

    def _get_valid_index(self, max_index, prompt):
        while True:
            try:
                idx = int(input(prompt))
                if 1 <= idx <= max_index:
                    return idx
                print(f"Введіть число від 1 до {max_index}.")
            except ValueError:
                print("Введіть коректне ціле число.")


class Algorithm:
    @staticmethod
    def greedy(servers: List[Dict], requests: List[Dict]) -> Tuple[Dict, float]:
        server_states = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
        C = 0

        r = [(j, requests[j]["c"] / requests[j]["d"]) for j in range(len(requests))]
        L = sorted(r, key=lambda x: x[1], reverse=True)

        for j, _ in L:
            best_server = -1
            min_remain = float('inf')

            for i, server in server_states.items():
                if (server["n"] < servers[i]["K"] and
                        server["used_volume"] + requests[j]["d"] <= servers[i]["D"]):
                    remain = servers[i]["D"] - (server["used_volume"] + requests[j]["d"])
                    if remain < min_remain:
                        min_remain = remain
                        best_server = i

            if best_server != -1:
                server_states[best_server]["R"].add(j)
                server_states[best_server]["n"] += 1
                server_states[best_server]["used_volume"] += requests[j]["d"]
                server_states[best_server]["C"] += requests[j]["c"]
                C += requests[j]["c"]

        return server_states, C

    @staticmethod
    def genetic(servers: List[Dict], requests: List[Dict], population_size: int = 50, mutation_rate: float = 0.1,
                generations: int = 100, max_stagnant: int = 20) -> Tuple[Dict, float]:
        if not servers or not requests:
            raise ValueError("No servers or requests available.")

        # Initial population with heuristic seeding
        population = Algorithm._initialize_population(servers, requests, population_size)
        best_fitness = float('-inf')
        best_solution = None
        stagnant_count = 0

        for gen in range(generations):
            new_population = []
            for _ in range(population_size // 2):
                # Tournament Selection
                parent1 = Algorithm._tournament_selection(population, servers, requests)
                parent2 = Algorithm._tournament_selection(population, servers, requests)

                # Uniform Crossover
                child1, child2 = Algorithm._uniform_crossover(parent1, parent2, servers, requests)

                # Adaptive Mutation
                mutation_rate_adaptive = mutation_rate * (1 - gen / generations)
                child1 = Algorithm._mutate(child1, servers, requests, mutation_rate_adaptive)
                child2 = Algorithm._mutate(child2, servers, requests, mutation_rate_adaptive)

                new_population.extend([child1, child2])

            # Elitism: Carry over the best solution
            best_current = max(population, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))
            new_population.append(best_current)

            population = new_population[:population_size]

            # Update best solution
            current_best_fitness = Algorithm._calculate_fitness(best_current, requests, servers)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = best_current
                stagnant_count = 0
            else:
                stagnant_count += 1

            if stagnant_count >= max_stagnant:
                break

        return best_solution, best_fitness

    @staticmethod
    def _initialize_population(servers: List[Dict], requests: List[Dict], population_size: int) -> List[Dict]:
        population = []
        for _ in range(population_size):
            solution = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
            for req_idx in range(len(requests)):
                server_idx = random.randint(0, len(servers) - 1)
                solution[server_idx]["R"].add(req_idx)
            population.append(Algorithm._repair_solution(solution, servers, requests))
        return population

    @staticmethod
    def _tournament_selection(population: List[Dict], servers: List[Dict], requests: List[Dict],
                              tournament_size: int = 3) -> Dict:
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))

    @staticmethod
    def _uniform_crossover(parent1: Dict, parent2: Dict, servers: List[Dict], requests: List[Dict]) -> Tuple[
        Dict, Dict]:
        child1 = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
        child2 = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
        for req_idx in range(len(requests)):
            if random.random() < 0.5:
                for s in range(len(servers)):
                    if req_idx in parent1[s]["R"]:
                        child1[s]["R"].add(req_idx)
                    if req_idx in parent2[s]["R"]:
                        child2[s]["R"].add(req_idx)
            else:
                for s in range(len(servers)):
                    if req_idx in parent2[s]["R"]:
                        child1[s]["R"].add(req_idx)
                    if req_idx in parent1[s]["R"]:
                        child2[s]["R"].add(req_idx)
        return Algorithm._repair_solution(child1, servers, requests), Algorithm._repair_solution(child2, servers,
                                                                                                 requests)

    @staticmethod
    def _mutate(solution: Dict, servers: List[Dict], requests: List[Dict], mutation_rate: float) -> Dict:
        if random.random() < mutation_rate:
            server1, server2 = random.sample(range(len(servers)), 2)
            if solution[server1]["R"] and solution[server2]["R"]:
                req1 = random.choice(list(solution[server1]["R"]))
                req2 = random.choice(list(solution[server2]["R"]))
                solution[server1]["R"].remove(req1)
                solution[server2]["R"].remove(req2)
                solution[server1]["R"].add(req2)
                solution[server2]["R"].add(req1)
        return Algorithm._repair_solution(solution, servers, requests)

    @staticmethod
    def _repair_solution(solution: Dict, servers: List[Dict], requests: List[Dict]) -> Dict:
        for s in solution:
            solution[s]["n"] = len(solution[s]["R"])
            solution[s]["used_volume"] = sum(requests[r]["d"] for r in solution[s]["R"])
            while (solution[s]["n"] > servers[s]["K"] or solution[s]["used_volume"] > servers[s]["D"]):
                if solution[s]["R"]:
                    worst_idx = min(solution[s]["R"], key=lambda x: requests[x]["c"] / requests[x]["d"])
                    solution[s]["R"].remove(worst_idx)
                    solution[s]["n"] -= 1
                    solution[s]["used_volume"] -= requests[worst_idx]["d"]
            solution[s]["C"] = sum(requests[r]["c"] for r in solution[s]["R"])
        return solution

    @staticmethod
    def _calculate_fitness(solution: Dict, requests: List[Dict], servers: List[Dict]) -> float:
        total_profit = sum(s["C"] for s in solution.values())
        penalty = 0
        for s in solution:
            if solution[s]["n"] > servers[s]["K"]:
                penalty += (solution[s]["n"] - servers[s]["K"]) * 100  # Arbitrary penalty
            if solution[s]["used_volume"] > servers[s]["D"]:
                penalty += (solution[s]["used_volume"] - servers[s]["D"]) * 100
        return total_profit - penalty

    @staticmethod
    def _survive(population: List[Dict], servers: List[Dict], requests: List[Dict]) -> List[Dict]:
        return sorted(population, key=lambda x: Algorithm._calculate_fitness(x, requests, servers), reverse=True)


class MenuController:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def display_menu(self):
        while True:
            print("""
Меню:
1 – Генерація випадкових запитів і серверів
2 – Завантаження серверів і запитів з файлу
3 – Редагування даних про сервери або запити
4 – Запуск алгоритмів розподілу запитів між серверами
5 – Збереження поточного списку серверів і запитів у файл
6 – Вибір експерименту
7 – Введення серверів і запитів вручну через консоль
0 – Завершення роботи програми
            """)
            option = input("Введіть опцію: ")

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
                print("\nВиберіть алгоритм:")
                print("1 – Жадібний алгоритм")
                print("2 – Генетичний алгоритм")
                algo_choice = input("Ваш вибір: ")
                if algo_choice == "1":
                    result, profit = Algorithm.greedy(self.data_manager.servers, self.data_manager.requests)
                    self._display_results(result, profit, "Жадібний алгоритм")
                elif algo_choice == "2":
                    result, profit = Algorithm.genetic(self.data_manager.servers, self.data_manager.requests)
                    self._display_results(result, profit, "Генетичний алгоритм")
                else:
                    print("Невірний вибір алгоритму.")
            elif option == "5":
                self.data_manager.save_data()
            elif option == "6":
                self._display_experiment_menu()
            elif option == "7":
                self.data_manager.manual_input()
            elif option == "0":
                print("Завершення роботи...")
                break
            else:
                print("Невірна опція, спробуйте ще раз.")

    def _display_experiment_menu(self):
        print("\nПідменю експериментів:")
        print("1 – Визначення параметра умови завершення роботи генетичного алгоритму")
        print("2 – Дослідження впливу параметрів генетичного алгоритму на його ефективність")
        print("3 – Визначення впливу параметрів задачі на точність та час роботи алгоритмів")
        choice = input("Виберіть номер експерименту: ")

        if choice == "1":
            self._experiment_1()
        elif choice == "2":
            self._experiment_2()
        elif choice == "3":
            self._experiment_3()
        else:
            print("Невірний вибір експерименту.")

    def _experiment_1(self):
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        generations_list = [100, 500, 1000]
        stagnant_list = [10, 20, 50]
        results = {}

        for gen in generations_list:
            for stagnant in stagnant_list:
                times = []
                profits = []
                for _ in range(5):  # 5 повторень для стабільності
                    start_time = time.time()
                    _, profit = Algorithm.genetic(self.data_manager.servers, self.data_manager.requests,
                                                  population_size=10, mutation_rate=0.5, max_stagnant=stagnant)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    profits.append(profit)
                avg_time = sum(times) / len(times)
                avg_profit = sum(profits) / len(profits)
                results[(gen, stagnant)] = {"time": avg_time, "profit": avg_profit}

        # Визначення оптимальної комбінації
        optimal = max(results.items(), key=lambda x: x[1]["profit"] / x[1]["time"])
        print("\nРезультати експерименту 1:")
        for (gen, stagnant), data in results.items():
            print(f"Покоління: {gen}, Незмінні рекорди: {stagnant}, "
                  f"Середній прибуток: {data['profit']:.2f}, "
                  f"Середній час: {data['time']:.4f} с")
        print(f"Оптимальна комбінація: Покоління={optimal[0][0]}, "
              f"Незмінні рекорди={optimal[0][1]}, "
              f"Баланс (прибуток/час): {optimal[1]['profit'] / optimal[1]['time']:.2f}")

    def _experiment_2(self):
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        n = len(self.data_manager.requests)
        c = sum(r["c"] for r in self.data_manager.requests) / n
        delta_c = c * 0.1  # 10% від середнього прибутку

        profits = []
        times = []

        for i in range(10):
            temp_requests = [
                {"id": r["id"], "d": r["d"], "c": max(1, r["c"] + random.uniform(-delta_c, delta_c))}
                for r in self.data_manager.requests
            ]
            start_time = time.time()
            _, profit = Algorithm.genetic(self.data_manager.servers, temp_requests, population_size=10,
                                          mutation_rate=0.5, max_stagnant=10)
            end_time = time.time()
            profits.append(profit)
            times.append(end_time - start_time)

        avg_profit = sum(profits) / 10
        avg_time = sum(times) / 10

        # Різниці критеріїв (відносно середнього)
        profit_diffs = [p - avg_profit for p in profits]

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), profit_diffs, label="Різниця критеріїв", color="green")
        plt.axhline(y=0, color="red", linestyle="--", label="Середній прибуток")
        plt.xlabel("Номер задачі")
        plt.ylabel("Різниця прибутку")
        plt.title("Різниці критеріїв для задач")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Середній прибуток: {avg_profit}")
        print(f"Середній час роботи: {avg_time} секунд")

    def _experiment_3(self):
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        n_servers_list = [2, 4, 6, 8, 10]
        n_requests_list = [5, 10, 15, 20, 25]
        greedy_profits = []
        greedy_times = []
        genetic_profits = []
        genetic_times = []

        for n_s in n_servers_list:
            temp_servers = [
                {"id": i + 1, "K": random.randint(1, 5), "D": random.randint(100, 500), "used_requests": 0,
                 "used_volume": 0}
                for i in range(n_s)
            ]
            profits_g = []
            times_g = []
            profits_ge = []
            times_ge = []
            for n_r in n_requests_list:
                temp_requests = [
                    {"id": j + 1, "d": random.randint(10, 100), "c": random.randint(10, 100)}
                    for j in range(n_r)
                ]

                # Жадібний алгоритм
                start_time = time.time()
                _, profit = Algorithm.greedy(temp_servers, temp_requests)
                end_time = time.time()
                profits_g.append(profit)
                times_g.append(end_time - start_time)

                # Генетичний алгоритм
                start_time = time.time()
                _, profit = Algorithm.genetic(temp_servers, temp_requests, population_size=10, mutation_rate=0.5,
                                              max_stagnant=10)
                end_time = time.time()
                profits_ge.append(profit)
                times_ge.append(end_time - start_time)

            # Середні значення для кожної кількості серверів
            greedy_profits.append(sum(profits_g) / len(profits_g))
            greedy_times.append(sum(times_g) / len(times_g))
            genetic_profits.append(sum(profits_ge) / len(profits_ge))
            genetic_times.append(sum(times_ge) / len(times_ge))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(n_servers_list, greedy_profits, label="Жадібний", color="blue")
        plt.plot(n_servers_list, genetic_profits, label="Генетичний", color="orange")
        plt.xlabel("Кількість серверів")
        plt.ylabel("Середній прибуток")
        plt.title("Вплив кількості серверів на прибуток")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(n_servers_list, greedy_times, label="Жадібний", color="blue")
        plt.plot(n_servers_list, genetic_times, label="Генетичний", color="orange")
        plt.xlabel("Кількість серверів")
        plt.ylabel("Середній час (с)")
        plt.title("Вплив кількості серверів на час роботи")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _display_results(self, result: Dict, profit: float, algo_name: str):
        print(f"\nРезультати {algo_name}:")
        print(f"Загальний прибуток: {profit}")
        for i, s in result.items():
            server = self.data_manager.servers[i]
            print(f"Сервер {i + 1} (K={server['K']}, D={server['D']}): "
                  f"Використано запитів {s['n']}/{server['K']}, "
                  f"Обсяг {s['used_volume']}/{server['D']}, Запити {list(s['R'])}")


if __name__ == "__main__":
    print("Примітка: Для графіків потрібна бібліотека matplotlib (встановіть через 'pip install matplotlib').")
    data_manager = DataManager()
    controller = MenuController(data_manager)
    controller.display_menu()