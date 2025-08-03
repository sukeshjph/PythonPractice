my_tuple = (10, 20, 30)
# print(my_tuple[2])

# newTuple = (5, )

# print(newTuple)

a, b, c  = my_tuple
print(a)
print(b)
print(c)

print(my_tuple.count(10))
print(my_tuple.index(10))

# ===== ADVANCED TUPLE EXAMPLES =====
print("\n=== ADVANCED TUPLE EXAMPLES ===")

# 1. Tuple Unpacking - Advanced Patterns
print("\n1. Advanced Tuple Unpacking:")
# Basic unpacking
coordinates = (10, 20, 30)
x, y, z = coordinates
print(f"Coordinates: x={x}, y={y}, z={z}")

# Extended unpacking with *
numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
first, second, *middle, last = numbers
print(f"First: {first}, Second: {second}, Middle: {middle}, Last: {last}")

# Ignoring values with _
person = ("John", "Doe", 30, "Engineer", "New York")
first_name, last_name, age, *_ = person
print(f"Name: {first_name} {last_name}, Age: {age}")

# 2. Nested Tuples
print("\n2. Nested Tuples:")
# Tuple of tuples
matrix = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
print(f"Matrix: {matrix}")
print(f"Element at [1][2]: {matrix[1][2]}")

# Tuple with mixed data types
student_record = ("Alice", 20, ("Math", 95), ("Science", 88))
name, age, (math_subject, math_grade), (science_subject, science_grade) = student_record
print(f"Student: {name}, Age: {age}")
print(f"{math_subject}: {math_grade}, {science_subject}: {science_grade}")

# 3. Tuple Methods and Operations
print("\n3. Tuple Methods and Operations:")
mixed_tuple = (1, 2, 2, 3, 2, 4, 5, 2)

# Count occurrences
count_2 = mixed_tuple.count(2)
print(f"Number of 2s: {count_2}")

# Find index
index_3 = mixed_tuple.index(3)
print(f"Index of 3: {index_3}")

# Find index with start parameter
index_2_after_2 = mixed_tuple.index(2, 2)
print(f"Index of 2 after position 2: {index_2_after_2}")

# 4. Tuple Comprehension (Generator Expression)
print("\n4. Tuple Comprehension:")
# Create tuple with generator expression
squares = tuple(x**2 for x in range(1, 6))
print(f"Squares: {squares}")

# Tuple with conditional logic
even_squares = tuple(x**2 for x in range(1, 11) if x % 2 == 0)
print(f"Even squares: {even_squares}")

# 5. Tuple vs List Performance
print("\n5. Tuple vs List Performance:")
import time

# Create large tuple and list
large_tuple = tuple(range(1000000))
large_list = list(range(1000000))

# Performance comparison for iteration
start_time = time.time()
for item in large_tuple:
    pass
tuple_time = time.time() - start_time

start_time = time.time()
for item in large_list:
    pass
list_time = time.time() - start_time

print(f"Tuple iteration time: {tuple_time:.4f} seconds")
print(f"List iteration time: {list_time:.4f} seconds")
print(f"Tuple is {list_time/tuple_time:.1f}x faster for iteration")

# 6. Tuple as Dictionary Keys
print("\n6. Tuple as Dictionary Keys:")
# Tuples are immutable, so they can be dictionary keys
coordinates_dict = {
    (0, 0): "Origin",
    (1, 1): "Point A",
    (2, 2): "Point B",
    (-1, -1): "Point C"
}

print(f"Coordinate dictionary: {coordinates_dict}")
print(f"Value at (1, 1): {coordinates_dict[(1, 1)]}")

# Complex tuple keys
student_grades = {
    ("Alice", "Math"): 95,
    ("Alice", "Science"): 88,
    ("Bob", "Math"): 87,
    ("Bob", "Science"): 92
}

print(f"Student grades: {student_grades}")
print(f"Alice's Math grade: {student_grades[('Alice', 'Math')]}")

# 7. Tuple Packing and Unpacking in Functions
print("\n7. Tuple Packing and Unpacking in Functions:")
def get_coordinates():
    return (10, 20, 30)

def get_person_info():
    return ("John", "Doe", 30, "Engineer")

# Unpacking return values
x, y, z = get_coordinates()
print(f"Coordinates: ({x}, {y}, {z})")

name, surname, age, profession = get_person_info()
print(f"Person: {name} {surname}, {age} years old, {profession}")

# 8. Tuple with Named Fields (Named Tuple Alternative)
print("\n8. Tuple with Named Fields:")
from collections import namedtuple

# Create named tuple
Person = namedtuple('Person', ['name', 'age', 'city'])
person1 = Person("Alice", 25, "New York")
person2 = Person("Bob", 30, "Los Angeles")

print(f"Person 1: {person1}")
print(f"Person 1 name: {person1.name}")
print(f"Person 1 age: {person1.age}")

# Convert to regular tuple
person_tuple = tuple(person1)
print(f"As regular tuple: {person_tuple}")

# 9. Tuple Slicing and Advanced Indexing
print("\n9. Tuple Slicing and Advanced Indexing:")
data_tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Basic slicing
first_three = data_tuple[:3]
last_three = data_tuple[-3:]
middle = data_tuple[3:7]

print(f"First three: {first_three}")
print(f"Last three: {last_three}")
print(f"Middle: {middle}")

# Step slicing
every_second = data_tuple[::2]
reverse = data_tuple[::-1]

print(f"Every second: {every_second}")
print(f"Reverse: {reverse}")

# 10. Tuple Concatenation and Repetition
print("\n10. Tuple Concatenation and Repetition:")
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

# Concatenation
combined = tuple1 + tuple2
print(f"Combined: {combined}")

# Repetition
repeated = tuple1 * 3
print(f"Repeated: {repeated}")

# 11. Tuple Comparison
print("\n11. Tuple Comparison:")
tuple_a = (1, 2, 3)
tuple_b = (1, 2, 4)
tuple_c = (1, 2, 3)

print(f"tuple_a == tuple_c: {tuple_a == tuple_c}")
print(f"tuple_a < tuple_b: {tuple_a < tuple_b}")
print(f"tuple_b > tuple_a: {tuple_b > tuple_a}")

# Lexicographic comparison
words = ("apple", "banana", "cherry")
sorted_words = sorted(words)
print(f"Sorted words: {sorted_words}")

# 12. Tuple with Sets and Lists
print("\n12. Tuple with Sets and Lists:")
# Tuple containing different data structures
complex_tuple = (
    [1, 2, 3],  # List
    {4, 5, 6},  # Set
    (7, 8, 9),  # Tuple
    {"a": 1, "b": 2}  # Dictionary
)

print(f"Complex tuple: {complex_tuple}")

# Accessing and modifying mutable elements
complex_tuple[0].append(4)  # Modify the list
print(f"After modifying list: {complex_tuple}")

# 13. Tuple Memory Efficiency
print("\n13. Tuple Memory Efficiency:")
import sys

# Compare memory usage
tuple_small = (1, 2, 3, 4, 5)
list_small = [1, 2, 3, 4, 5]

tuple_size = sys.getsizeof(tuple_small)
list_size = sys.getsizeof(list_small)

print(f"Tuple size: {tuple_size} bytes")
print(f"List size: {list_size} bytes")
print(f"Tuple uses {tuple_size/list_size:.1%} of list memory")

# 14. Real-world Example: Database Records
print("\n14. Real-world Example - Database Records:")
# Simulate database records
employees = [
    ("E001", "Alice", "Johnson", 50000, "Engineering"),
    ("E002", "Bob", "Smith", 60000, "Marketing"),
    ("E003", "Charlie", "Brown", 55000, "Engineering"),
    ("E004", "Diana", "Wilson", 65000, "Sales")
]

# Process employee data
total_salary = sum(emp[3] for emp in employees)
avg_salary = total_salary / len(employees)

print(f"Total salary: ${total_salary:,}")
print(f"Average salary: ${avg_salary:,.2f}")

# Group by department
dept_salaries = {}
for emp in employees:
    dept = emp[4]
    salary = emp[3]
    if dept not in dept_salaries:
        dept_salaries[dept] = []
    dept_salaries[dept].append(salary)

for dept, salaries in dept_salaries.items():
    avg_dept_salary = sum(salaries) / len(salaries)
    print(f"{dept} average salary: ${avg_dept_salary:,.2f}")

# 15. Tuple with Lambda Functions
print("\n15. Tuple with Lambda Functions:")
# Tuple of lambda functions
operations = (
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y if y != 0 else "Error"
)

# Apply operations
a, b = 10, 5
for i, op in enumerate(operations):
    result = op(a, b)
    print(f"Operation {i+1}: {a} op {b} = {result}")

# 16. Tuple with Enumerate
print("\n16. Tuple with Enumerate:")
fruits = ("apple", "banana", "cherry", "date")

# Enumerate with tuple unpacking
for index, fruit in enumerate(fruits):
    print(f"Fruit {index + 1}: {fruit}")

# Create tuple of (index, value) pairs
indexed_fruits = tuple(enumerate(fruits))
print(f"Indexed fruits: {indexed_fruits}")

# 17. Tuple with Zip
print("\n17. Tuple with Zip:")
names = ("Alice", "Bob", "Charlie")
ages = (25, 30, 35)
cities = ("NYC", "LA", "Chicago")

# Zip multiple tuples
people = tuple(zip(names, ages, cities))
print(f"People: {people}")

# Unzip
unzipped_names, unzipped_ages, unzipped_cities = zip(*people)
print(f"Unzipped names: {unzipped_names}")
print(f"Unzipped ages: {unzipped_ages}")
print(f"Unzipped cities: {unzipped_cities}")

# 18. Tuple with Filter and Map
print("\n18. Tuple with Filter and Map:")
numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Filter even numbers
even_numbers = tuple(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_numbers}")

# Map to squares
squared_numbers = tuple(map(lambda x: x**2, numbers))
print(f"Squared numbers: {squared_numbers}")

# Combined filter and map
even_squares = tuple(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print(f"Even squares: {even_squares}")

# 19. Tuple with Reduce
print("\n19. Tuple with Reduce:")
from functools import reduce

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(f"Total: {total}")

# Find maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(f"Maximum: {maximum}")

# 20. Tuple with Any and All
print("\n20. Tuple with Any and All:")
test_tuple = (True, False, True, True)

# Check if any element is True
any_true = any(test_tuple)
print(f"Any True: {any_true}")

# Check if all elements are True
all_true = all(test_tuple)
print(f"All True: {all_true}")

# Practical example
grades = (85, 92, 78, 95, 88)
passing_grades = tuple(grade >= 80 for grade in grades)
print(f"Passing grades: {passing_grades}")
print(f"All passing: {all(passing_grades)}")
print(f"Any failing: {not all(passing_grades)}")