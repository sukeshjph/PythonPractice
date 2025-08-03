class Employee:
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

    def getSalary(self):
        return self.salary
    
    def getDetails(self):
        return {"name": self.name, "age": self.age, "salary": self.salary}
    

class Child(Employee):
    def __init__(self, name, age, salary,designation):
        super().__init__(name, age, salary)
        self.designation = designation

    def getSalary(self):
        return super().getSalary()+5000
    
    def getDetails(self):
        return {"name": self.name, "age": self.age, "salary": self.salary, "designation": self.designation}

employee = Employee("Sukesh",40, 34000)

print(employee.getDetails())
print(employee.getSalary())

childEm = Child("Sukesh", 40, 34000, "Developer")
print(childEm.getSalary())
print(childEm.getDetails())

