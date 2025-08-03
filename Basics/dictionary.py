# dict = {"name": "Alice", "age": 21, "city": "New York"}

# print(dict["name"])
# print(dict.keys())
# print(dict.values())
# print(dict.items())

# dict.clear()
# print(dict)

# dict["name"] = "Bob"
# print(dict.get("name1","Suku"))

# print(dict)

# person = dict()
# person["name"] = 'John'
# person["age"] = 20
# person["city"] = 'London'

# print(person)

# ===== ADVANCED DICTIONARY EXAMPLES =====
print("\n=== ADVANCED DICTIONARY EXAMPLES ===")

# 1. Nested Dictionaries
print("\n1. Nested Dictionaries:")
student = {
    "name": "Alice",
    "age": 20,
    "grades": {
        "math": 95,
        "science": 88,
        "english": 92
    },
    "address": {
        "street": "123 Main St",
        "city": "Boston",
        "zip": "02101"
    }
}

print(f"Student: {student['name']}")
print(f"Math grade: {student['grades']['math']}")
print(f"City: {student['address']['city']}")

studentWithMarks = {
    "Alice": 9,
    "Bob": 8,
    "Charlie": 6,
    "John": 5
}

# 2. Dictionary Comprehension from a list using range
print("\n2. Dictionary Comprehension:")
# # Create dictionary with squares
# squares = {x: x**2 for x in range(1, 6)}
# print(f"Squares: {squares}")

# cubes = {x: x**3 for x in range(1,10)}
# print(f'Cubes:{cubes}')

studentKeys = studentWithMarks.keys()
studentValues = studentWithMarks.values()

studentGradesMultiplierDict = {k:v**2 for (k, v) in zip(studentKeys, studentValues)}
studentGradesMultiplierDict2 = {k.upper():v**2 for (k, v) in zip(studentKeys, studentValues)}

for key, score in studentGradesMultiplierDict.items():
    print(f'{key}: {score}')

for key, score in studentGradesMultiplierDict2.items():
    print(f'{key}: {score}')







# Create dictionary with conditional logic
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# 3. Dictionary Methods - Advanced
print("\n3. Advanced Dictionary Methods:")
fruits = {"apple": 5, "banana": 3, "orange": 7, "grape": 2}

# setdefault() - sets default if key doesn't exist
fruits.setdefault("mango", 0)
fruits.setdefault("apple", 10)  # Won't change existing value
print(f"After setdefault: {fruits}")

# update() - merge dictionaries
more_fruits = {"kiwi": 4, "pear": 6}
fruits.update(more_fruits)
print(f"After update: {fruits}")

# 4. Dictionary Merging (Python 3.9+)
print("\n4. Dictionary Merging:")
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
dict3 = {"b": 5, "e": 6}  # Note: 'b' exists in both

# Method 1: Using | operator (Python 3.9+)
merged = dict1 | dict2 | dict3
print(f"Merged with |: {merged}")

# Method 2: Using ** unpacking
merged2 = {**dict1, **dict2, **dict3}
print(f"Merged with **: {merged2}")

merged3 = {**dict1, **{"j": 10}}
print(f"Merged with ** and dict: {merged3}")




# 5. Dictionary with Lists and Sets
print("\n5. Dictionary with Lists and Sets:")
class_info = {
    "students": ["Alice", "Bob", "Charlie", "Diana"],
    "subjects": {"math", "science", "english", "history"},
    "grades": {
        "Alice": [95, 88, 92],
        "Bob": [87, 91, 85],
        "Charlie": [92, 89, 94],
        "Diana": [88, 93, 90]
    }
}

print(f"Number of students: {len(class_info['students'])}")
print(f"Subjects: {class_info['subjects']}")
print(f"Alice's grades: {class_info['grades']['Alice']}")

# 6. Dictionary as Switch/Case Alternative
print("\n6. Dictionary as Switch/Case:")
def add(x, y): return x + y
def subtract(x, y): return x - y
def multiply(x, y): return x * y
def divide(x, y): return x / y if y != 0 else "Error: Division by zero"

operations = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide
}

# Using the dictionary as a switch
operation = "+"
x, y = 10, 5
result = operations[operation](x, y)
print(f"{x} {operation} {y} = {result}")

# 7. Dictionary with Default Values
print("\n7. Dictionary with Default Values:")
from collections import defaultdict

# defaultdict automatically creates default values
word_count = defaultdict(int)
sentence = "the quick brown fox jumps over the lazy dog"
for word in sentence.split():
    word_count[word] += 1

print(f"Word count: {dict(word_count)}")

# 8. Dictionary Sorting
print("\n8. Dictionary Sorting:")
scores = {"Alice": 95, "Bob": 87, "Charlie": 92, "Diana": 88}

# Sort by keys
sorted_by_name = dict(sorted(scores.items()))
print(f"Sorted by name: {sorted_by_name}")

# Sort by values (descending)
sorted_by_score = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
print(f"Sorted by score (descending): {sorted_by_score}")

# 9. Dictionary Filtering
print("\n9. Dictionary Filtering:")
prices = {"apple": 1.50, "banana": 0.75, "orange": 2.00, "grape": 3.50, "kiwi": 1.25}

# Filter expensive items (> $2.00)
expensive = {k: v for k, v in prices.items() if v > 2.00}
print(f"Expensive items: {expensive}")

# Filter by key length
long_names = {k: v for k, v in prices.items() if len(k) > 5}
print(f"Items with long names: {long_names}")

# 10. Real-world Example: Inventory System


print("\n10. Real-world Example - Inventory System:")
inventory = {
    "laptop": {"price": 999.99, "quantity": 5, "category": "electronics"},
    "book": {"price": 19.99, "quantity": 20, "category": "books"},
    "coffee": {"price": 4.99, "quantity": 50, "category": "beverages"},
    "headphones": {"price": 89.99, "quantity": 8, "category": "electronics"}
}

# Calculate total value
total_value = sum(item["price"] * item["quantity"] for item in inventory.values())
print(f"Total inventory value: ${total_value:.2f}")

# Find items by category
electronics = {k: v for k, v in inventory.items() if v["category"] == "electronics"}
print(f"Electronics: {list(electronics.keys())}")

# Find low stock items
low_stock = {k: v for k, v in inventory.items() if v["quantity"] < 10}
print(f"Low stock items: {list(low_stock.keys())}")
 
pricegreaterthan80 = {k:v for k, v in inventory.items() if v["price"] > 80}

print(f'Values great than 80: {list(pricegreaterthan80.keys())}')
