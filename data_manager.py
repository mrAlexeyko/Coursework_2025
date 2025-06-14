import random
import json

class DataManager:
    """Клас для керування даними серверів і запитів."""
    def __init__(self):
        self.servers = []
        self.requests = []

    def generate_data(self):
        """Генерація випадкових даних для серверів і запитів."""
        n_servers = self._get_positive_int("Введіть кількість серверів: ")
        n_requests = self._get_positive_int("Введіть кількість запитів: ")

        min_K = self._get_positive_int("Введіть мінімальне значення для K: ")
        max_K = self._get_positive_int(f"Введіть максимальне значення для K (>{min_K}): ", min_K)
        min_D = self._get_positive_int("Введіть мінімальне значення для D: ")
        max_D = self._get_positive_int(f"Введіть максимальне значення для D (>{min_D}): ", min_D)
        min_d = self._get_positive_int("Введіть мінімальне значення для d: ")
        max_d = self._get_positive_int(f"Введіть максимальне значення для d (>{min_d}): ", min_d)
        min_c = self._get_positive_int("Введіть мінімальне значення для c: ")
        max_c = self._get_positive_int(f"Введіть максимальне значення для c (>{min_c}): ", min_c)

        self.servers = [{"id": i + 1, "K": random.randint(min_K, max_K), "D": random.randint(min_D, max_D),
                         "used_requests": 0, "used_volume": 0} for i in range(n_servers)]
        self.requests = [{"id": j + 1, "d": random.randint(min_d, max_d), "c": random.randint(min_c, max_c)}
                         for j in range(n_requests)]

        print("\nЗгенеровані сервери:", *self.servers, sep="\n")
        print("\nЗгенеровані запити:", *self.requests, sep="\n")

    def load_data(self):
        """Завантаження даних із JSON-файлу."""
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
            print(f"Помилка завантаження даних: {e}")

    def save_data(self):
        """Збереження даних у JSON-файл."""
        if not self.servers or not self.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані.")
            return
        filename = input("Введіть назву JSON-файлу для збереження: ")
        try:
            with open(filename, 'w') as f:
                json.dump({"servers": self.servers, "requests": self.requests}, f, indent=2)
            print("Дані успішно збережено.")
        except Exception as e:
            print(f"Помилка збереження даних: {e}")

    def edit_data(self):
        """Редагування даних серверів або запитів."""
        if not self.servers and not self.requests:
            print("Помилка: Спочатку згенеруйте або завантажте дані.")
            return
        choice = input("Редагувати (1 - сервери, 2 - запити): ")
        if choice == "1" and self.servers:
            print(*[f"{i + 1}. {s}" for i, s in enumerate(self.servers)], sep="\n")
            idx = self._get_valid_index(len(self.servers), "Введіть номер сервера для редагування: ") - 1
            self.servers[idx]["K"] = self._get_positive_int("Нове значення для K: ")
            self.servers[idx]["D"] = self._get_positive_int("Нове значення для D: ")
        elif choice == "2" and self.requests:
            print(*[f"{i + 1}. {r}" for i, r in enumerate(self.requests)], sep="\n")
            idx = self._get_valid_index(len(self.requests), "Введіть номер запиту для редагування: ") - 1
            self.requests[idx]["d"] = self._get_positive_int("Нове значення для d: ")
            self.requests[idx]["c"] = self._get_positive_int("Нове значення для c: ")
        else:
            print("Неправильний вибір або дані відсутні.")

    def manual_input(self):
        """Ручне введення даних серверів і запитів."""
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
        """Отримання позитивного цілого числа від користувача."""
        while True:
            try:
                value = int(input(prompt))
                if value <= min_value:
                    print(f"Введіть число більше за {min_value}.")
                    continue
                return value
            except ValueError:
                print("Введіть коректне ціле число.")

    def _get_valid_index(self, max_index, prompt):
        """Отримання валідного індексу в заданому діапазоні."""
        while True:
            try:
                idx = int(input(prompt))
                if 1 <= idx <= max_index:
                    return idx
                print(f"Введіть число від 1 до {max_index}.")
            except ValueError:
                print("Введіть коректне ціле число.")