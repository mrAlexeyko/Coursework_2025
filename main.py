from data_manager import DataManager
from experiment import Experiment
from menu_controller import MenuController

if __name__ == "__main__":
    print("Примітка: Для графіків потрібна бібліотека matplotlib (встановіть через 'pip install matplotlib').")
    data_manager = DataManager()
    experiment = Experiment(data_manager)
    controller = MenuController(data_manager, experiment)
    controller.display_menu()