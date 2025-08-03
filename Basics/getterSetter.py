class Employee:
    def __init__(self, name, salary):
        self._name = name
        self._salary = salary


    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, new_name):
        self._name = new_name

    @name.deleter
    def name(self):
        del self._name
    


empl = Employee("Sukesh", 23455)

print(empl.name)
empl.name = "Rakesh"
print(empl.name)

del empl.name
print(empl._salary)