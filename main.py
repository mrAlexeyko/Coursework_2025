import json
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time


class DataManager:
    """Клас для управління даними серверів і запитів."""

    def __init__(self):
        self.servers = []
        self.requests = []

    def generate_data(self):
        """Генерує випадкові дані для серверів і запитів."""
        n_servers = self._get_positive_int("Введіть кількість серверів: ")
        n_requests = self._get_positive_int("Введіть кількість запитів: ")

        min_K = self._get_positive_int("Введіть мінімальне значення K: ")
        max_K = self._get_positive_int(f"Введіть максимальне значення K (>{min_K}): ", min_K)
        min_D = self._get_positive_int("Введіть мінімальне значення D: ")
        max_D = self._get_positive_int(f"Введіть максимальне значення D (>{min_D}): ", min_D)
        min_d = self._get_positive_int("Введіть мінімальне значення d: ")
        max_d = self._get_positive_int(f"Введіть максимальне значення d (>{min_d}): ", min_d)
        min_c = self._get_positive_int("Введіть мінімальне значення c: ")
        max_c = self._get_positive_int(f"Введіть максимальне значення c (>{min_c}): ", min_c)

        self.servers = [{"id": i + 1, "K": random.randint(min_K, max_K), "D": random.randint(min_D, max_D),
                         "used_requests": 0, "used_volume": 0} for i in range(n_servers)]
        self.requests = [{"id": j + 1, "d": random.randint(min_d, max_d), "c": random.randint(min_c, max_c)}
                         for j in range(n_requests)]

        print("\nЗгенеровані сервери:", *self.servers, sep="\n")
        print("\nЗгенеровані запити:", *self.requests, sep="\n")

    def load_data(self):
        """Завантажує дані з JSON-файлу."""
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
        """Зберігає дані у JSON-файл."""
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
        """Дозволяє редагувати дані серверів або запитів."""
        if not self.servers and not self.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані.")
            return
        choice = input("Редагувати (1 – сервери, 2 – запити): ")
        if choice == "1" and self.servers:
            print(*[f"{i + 1}. {s}" for i, s in enumerate(self.servers)], sep="\n")
            idx = self._get_valid_index(len(self.servers), "Введіть номер сервера для редагування: ") - 1
            self.servers[idx]["K"] = self._get_positive_int("Нове значення K: ")
            self.servers[idx]["D"] = self._get_positive_int("Нове значення D: ")
        elif choice == "2" and self.requests:
            print(*[f"{i + 1}. {r}" for i, r in enumerate(self.requests)], sep="\n")
            idx = self._get_valid_index(len(self.requests), "Введіть номер запиту для редагування: ") - 1
            self.requests[idx]["d"] = self._get_positive_int("Нове значення d: ")
            self.requests[idx]["c"] = self._get_positive_int("Нове значення c: ")
        else:
            print("Невірний вибір або відсутні дані.")

    def manual_input(self):
        """Дозволяє вручну ввести дані серверів і запитів."""
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

        print("\nВведені сервери:", *self.servers, sep="\n")
        print("\nВведені запити:", *self.requests, sep="\n")

    def _get_positive_int(self, prompt, min_value=0):
        """Отримує додатне ціле число від користувача."""
        while True:
            try:
                value = int(input(prompt))
                if value <= min_value:
                    print(f"Введіть число більше {min_value}.")
                    continue
                return value
            except ValueError:
                print("Введіть коректне ціле число.")

    def _get_valid_index(self, max_index, prompt):
        """Отримує коректний індекс у заданому діапазоні."""
        while True:
            try:
                idx = int(input(prompt))
                if 1 <= idx <= max_index:
                    return idx
                print(f"Введіть число від 1 до {max_index}.")
            except ValueError:
                print("Введіть коректне ціле число.")


class Algorithm:
    """Клас із методами жадібного та генетичного алгоритмів."""

    @staticmethod
    def greedy(servers: List[Dict], requests: List[Dict]) -> Tuple[Dict, float]:
        """Жадібний алгоритм для розподілу запитів між серверами."""
        server_states = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
        total_profit = 0

        sorted_requests = sorted(enumerate(requests), key=lambda x: x[1]["c"] / x[1]["d"], reverse=True)

        for j, _ in sorted_requests:
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
                total_profit += requests[j]["c"]

        return server_states, total_profit

    @staticmethod
    def genetic(servers: List[Dict], requests: List[Dict], population_size: int = 50, mutation_rate: float = 0.1,
                generations: int = 100, max_stagnant: int = 20) -> Tuple[Dict, float]:
        """Генетичний алгоритм для розподілу запитів між серверами."""
        if not servers or not requests:
            raise ValueError("Немає серверів або запитів.")

        population = Algorithm._initialize_population(servers, requests, population_size)
        best_fitness = float('-inf')
        best_solution = None
        stagnant_count = 0

        for _ in range(generations):
            new_population = []
            for _ in range(population_size // 2):
                parent1 = Algorithm._tournament_selection(population, servers, requests)
                parent2 = Algorithm._tournament_selection(population, servers, requests)
                child1, child2 = Algorithm._uniform_crossover(parent1, parent2, servers, requests)
                child1 = Algorithm._mutate(child1, servers, requests, mutation_rate)
                child2 = Algorithm._mutate(child2, servers, requests, mutation_rate)
                new_population.extend([child1, child2])

            best_current = max(population, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))
            new_population.append(best_current)
            population = new_population[:population_size]

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
        """Ініціалізує початкову популяцію."""
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
        """Вибирає найкращу особину турнірним відбором."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))

    @staticmethod
    def _uniform_crossover(parent1: Dict, parent2: Dict, servers: List[Dict], requests: List[Dict]) -> Tuple[
        Dict, Dict]:
        """Виконує рівномірний кросовер між двома батьками."""
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
        """Виконує мутацію розв’язку."""
        if random.random() < mutation_rate and len(servers) >= 2:
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
        """Виправляє розв’язок, щоб він відповідав обмеженням."""
        for s in solution:
            solution[s]["n"] = len(solution[s]["R"])
            solution[s]["used_volume"] = sum(requests[r]["d"] for r in solution[s]["R"])
            while solution[s]["n"] > servers[s]["K"] or solution[s]["used_volume"] > servers[s]["D"]:
                if solution[s]["R"]:
                    worst_idx = min(solution[s]["R"], key=lambda x: requests[x]["c"] / requests[x]["d"])
                    solution[s]["R"].remove(worst_idx)
                    solution[s]["n"] -= 1
                    solution[s]["used_volume"] -= requests[worst_idx]["d"]
            solution[s]["C"] = sum(requests[r]["c"] for r in solution[s]["R"])
        return solution

    @staticmethod
    def _calculate_fitness(solution: Dict, requests: List[Dict], servers: List[Dict]) -> float:
        """Обчислює придатність розв’язку."""
        total_profit = sum(s["C"] for s in solution.values())
        penalty = 0
        for s in solution:
            if solution[s]["n"] > servers[s]["K"]:
                penalty += (solution[s]["n"] - servers[s]["K"]) * 100
            if solution[s]["used_volume"] > servers[s]["D"]:
                penalty += (solution[s]["used_volume"] - servers[s]["D"]) * 100
        return total_profit - penalty


class Experiment:
    """Клас для проведення експериментів із алгоритмами."""
    """Клас для проведення експериментів із алгоритмами."""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def experiment_1(self):
        """Експеримент 1: Визначення оптимальних параметрів генетичного алгоритму з опціональним збереженням у JSON."""
        if not self.data_manager.servers or not self.data_manager.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані (опції 1 або 2).")
            return

        generations_list = [100, 500, 1000]
        stagnant_list = [10, 20, 50]
        results = {}

        for gen in generations_list:
            for stagnant in stagnant_list:
                times, profits = [], []
                for _ in range(5):  # 5 повторень для стабільності
                    start_time = time.time()
                    _, profit = Algorithm.genetic(self.data_manager.servers, self.data_manager.requests,
                                                  generations=gen, max_stagnant=stagnant)
                    times.append(time.time() - start_time)
                    profits.append(profit)
                avg_time = sum(times) / 5
                avg_profit = sum(profits) / 5
                balance = avg_profit / avg_time if avg_time > 0 else 0
                results[(gen, stagnant)] = {"time": avg_time, "profit": avg_profit, "balance": balance}

        # Визначення оптимальної комбінації
        optimal = max(results.items(), key=lambda x: x[1]["balance"])
        print("\nРезультати експерименту 1:")
        for (gen, stagnant), data in results.items():
            print(f"Покоління: {gen}, Незмінні рекорди: {stagnant}, "
                  f"Середній прибуток: {data['profit']:.2f}, Середній час: {data['time']:.4f} с, "
                  f"Баланс (прибуток/час): {data['balance']:.2f}")
        print(f"Оптимальна комбінація: Покоління={optimal[0][0]}, "
              f"Незмінні рекорди={optimal[0][1]}, "
              f"Баланс (прибуток/час): {optimal[1]['balance']:.2f}")

        # Опціональне збереження у JSON
        save_choice = input("Бажаєте зберегти результати в JSON-файл? (1 - так, 2 - ні): ")
        if save_choice == "1":
            filename = input("Введіть ім’я JSON-файлу для збереження: ")
            try:
                # Перетворення результатів у більш наглядну структуру
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
                print(f"Результати успішно збережено у {filename}")
            except Exception as e:
                print(f"Помилка збереження: {e}")

    def experiment_2(self):
        """Експеримент 2: Вплив варіації прибутку на генетичний алгоритм."""
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
        plt.plot(range(1, 11), profit_diffs, label="Різниця критеріїв", color="green")
        plt.axhline(y=0, color="red", linestyle="--", label="Середній прибуток")
        plt.xlabel("Номер задачі")
        plt.ylabel("Різниця прибутку")
        plt.title("Різниці критеріїв для задач")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Середній прибуток: {avg_profit:.2f}")
        print(f"Середній час роботи: {avg_time:.4f} секунд")

    def experiment_3(self):
        """Експеримент 3: Вплив розміру задачі на алгоритми."""
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


class MenuController:
    """Клас для управління меню програми."""

    def __init__(self, data_manager: DataManager, experiment: Experiment):
        self.data_manager = data_manager
        self.experiment = experiment

    def display_menu(self):
        """Відображає головне меню та обробляє вибір користувача."""
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
                algo_choice = input("\nВиберіть алгоритм:\n1 – Жадібний алгоритм\n2 – Генетичний алгоритм\nВаш вибір: ")
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
        """Відображає підменю експериментів."""
        choice = input("\nПідменю експериментів:\n"
                       "1 – Визначення параметра умови завершення роботи генетичного алгоритму\n"
                       "2 – Дослідження впливу параметрів генетичного алгоритму на його ефективність\n"
                       "3 – Визначення впливу параметрів задачі на точність та час роботи алгоритмів\n"
                       "Виберіть номер експерименту: ")
        if choice == "1":
            self.experiment.experiment_1()
        elif choice == "2":
            self.experiment.experiment_2()
        elif choice == "3":
            self.experiment.experiment_3()
        else:
            print("Невірний вибір експерименту.")

    def _display_results(self, result: Dict, profit: float, algo_name: str):
        """Відображає результати роботи алгоритму."""
        print(f"\nРезультати {algo_name}:")
        print(f"Загальний прибуток: {profit}")
        for i, s in result.items():
            server = self.data_manager.servers[i]
            print(f"Сервер {i + 1} (K={server['K']}, D={server['D']}): "
                  f"Використано запитів {s['n']}/{server['K']}, Обсяг {s['used_volume']}/{server['D']}, "
                  f"Запити {list(s['R'])}")


if __name__ == "__main__":
    print("Примітка: Для графіків потрібна бібліотека matplotlib (встановіть через 'pip install matplotlib').")
    data_manager = DataManager()
    experiment = Experiment(data_manager)
    controller = MenuController(data_manager, experiment)
    controller.display_menu()