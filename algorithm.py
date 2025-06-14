import random
from typing import List, Dict, Tuple


class Algorithm:
    """Клас, що містить жадібний та генетичний алгоритми."""
    @staticmethod
    def greedy(servers: List[Dict], requests: List[Dict]) -> Tuple[Dict, float]:
        """Жадібний алгоритм для розподілу запитів між серверами."""
        server_states = {
            s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0} for s in servers}
        total_profit = 0

        sorted_requests = sorted(
            enumerate(requests), key=lambda x: x[1]["c"] / x[1]["d"], reverse=True)

        for j, _ in sorted_requests:
            best_server = -1
            min_remain = float('inf')
            for i, server in server_states.items():
                if (server["n"] < servers[i]["K"] and
                        server["used_volume"] + requests[j]["d"] <= servers[i]["D"]):
                    remain = servers[i]["D"] - \
                        (server["used_volume"] + requests[j]["d"])
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
            raise ValueError("Сервери або запити відсутні.")

        population = Algorithm._initialize_population(
            servers, requests, population_size)
        best_fitness = float('-inf')
        best_solution = None
        stagnant_count = 0

        for _ in range(generations):
            new_population = []
            for _ in range(population_size // 2):
                parent1 = Algorithm._tournament_selection(
                    population, servers, requests)
                parent2 = Algorithm._tournament_selection(
                    population, servers, requests)
                # parent1 = Algorithm._phenotypic_outbreeding_selection(
                #     population, servers, requests)
                # parent2 = Algorithm._phenotypic_outbreeding_selection(
                #     population, servers, requests)
                child1, child2 = Algorithm._uniform_crossover(
                    parent1, parent2, servers, requests)
                child1 = Algorithm._mutate(
                    child1, servers, requests, mutation_rate)
                child2 = Algorithm._mutate(
                    child2, servers, requests, mutation_rate)
                new_population.extend([child1, child2])

            best_current = max(
                population, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))
            new_population.append(best_current)
            population = new_population[:population_size]

            current_best_fitness = Algorithm._calculate_fitness(
                best_current, requests, servers)
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
        """Ініціалізація популяції для генетичного алгоритму."""
        population = []
        for _ in range(population_size):
            solution = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0}
                        for s in servers}
            for req_idx in range(len(requests)):
                server_idx = random.randint(0, len(servers) - 1)
                solution[server_idx]["R"].add(req_idx)
            population.append(Algorithm._repair_solution(
                solution, servers, requests))
        return population

    @staticmethod
    def _tournament_selection(population: List[Dict], servers: List[Dict], requests: List[Dict], tournament_size: int = 3) -> Dict:
        """Вибір найкращого індивідуума за допомогою турнірного відбору."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: Algorithm._calculate_fitness(x, requests, servers))

    @staticmethod
    def _phenotypic_outbreeding_selection(population: List[Dict], servers: List[Dict], requests: List[Dict]) -> Tuple[Dict, Dict]:
        """Вибір двох батьків з максимальною різницею в ЦФ для аутбридингу."""
        if len(population) < 2:
            raise ValueError("Популяція занадто мала для аутбридингу.")

        fitness_values = [(ind, Algorithm._calculate_fitness(
            ind, requests, servers)) for ind in population]
        fitness_values.sort(key=lambda x: x[1])

        parent1 = fitness_values[0][0]
        parent2 = fitness_values[-1][0]

        return parent1.copy(), parent2.copy()

    @staticmethod
    def _uniform_crossover(parent1: Dict, parent2: Dict, servers: List[Dict], requests: List[Dict]) -> Tuple[Dict, Dict]:
        """Виконання рівномірного кросоверу між двома батьками."""
        child1 = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0}
                  for s in servers}
        child2 = {s["id"] - 1: {"R": set(), "n": 0, "used_volume": 0, "C": 0}
                  for s in servers}
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
        return Algorithm._repair_solution(child1, servers, requests), Algorithm._repair_solution(child2, servers, requests)

    @staticmethod
    def _mutate(solution: Dict, servers: List[Dict], requests: List[Dict], mutation_rate: float) -> Dict:
        """Мутація рішення."""
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
        """Виправлення рішення для відповідності обмеженням."""
        for s in solution:
            solution[s]["n"] = len(solution[s]["R"])
            solution[s]["used_volume"] = sum(
                requests[r]["d"] for r in solution[s]["R"])
            while solution[s]["n"] > servers[s]["K"] or solution[s]["used_volume"] > servers[s]["D"]:
                if solution[s]["R"]:
                    worst_idx = min(
                        solution[s]["R"], key=lambda x: requests[x]["c"] / requests[x]["d"])
                    solution[s]["R"].remove(worst_idx)
                    solution[s]["n"] -= 1
                    solution[s]["used_volume"] -= requests[worst_idx]["d"]
            solution[s]["C"] = sum(requests[r]["c"] for r in solution[s]["R"])
        return solution

    @staticmethod
    def _calculate_fitness(solution: Dict, requests: List[Dict], servers: List[Dict]) -> float:
        """Обчислення придатності рішення."""
        total_profit = sum(s["C"] for s in solution.values())
        penalty = 0
        for s in solution:
            if solution[s]["n"] > servers[s]["K"]:
                penalty += (solution[s]["n"] - servers[s]["K"]) * 100
            if solution[s]["used_volume"] > servers[s]["D"]:
                penalty += (solution[s]["used_volume"] - servers[s]["D"]) * 100
        return total_profit - penalty
