class Vehicle:
    def __init__(self,type_v,name,speed):
        self.__type = type_v
        self.__name = name
        self.__speed = speed
        print(f"{self.__type} added: {self.__name}")

    def classify(self):
        if self.__speed > 100:
            return "High-speed Vehicle"
        return "Normal Vehicle"

    def get_type(self):
        return self.__type

    def get_name(self):
        return self.__name

    def get_speed(self):
        return self.__speed

    def show_info(self):
        pass

class Car(Vehicle):
    def __init__(self, type_v, name, speed):
        super().__init__(type_v, name, speed)
        self.show_info()

    def show_info(self):
        print(f"Type: {self.get_type()}, Name: {self.get_name()}, Speed: {self.get_speed()}, Predicted: {self.classify()}")

class Bike(Vehicle):
    def __init__(self, type_v, name, speed):
        super().__init__(type_v, name, speed)

    def show_info(self):
        print(f"Type: {self.get_type()}, Name: {self.get_name()}, Speed: {self.get_speed()}, Predicted: {self.classify()}")


c1 = Car("Car","Toyota",150)

