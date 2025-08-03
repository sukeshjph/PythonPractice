# nums = [1,2,3,4]
# squares = [x*x for x in nums if x%2 == 0]
# print(squares)

arr = [1,2,55,6]
extendList = [53,23,34]

# arr.append(99)

# print(arr)
# print(",".join(str(x) for x in arr))

# popped = arr.pop()

# print(popped)
# print(arr)

# arr.extend(extendList)
# print(arr)

# arr.insert(1,78)

# print(arr)

# arr.sort()

# arr.reverse()

print(arr)

# ===== ADVANCED LIST COMPREHENSION EXAMPLES =====
print("\n=== ADVANCED LIST COMPREHENSION EXAMPLES ===")

# 1. Nested List Comprehensions
print("\n1. Nested List Comprehensions:")
# Create a 3x3 matrix
matrix = [[i+j for j in range(3)] for i in range(0, 9, 3)]
print(f"3x3 Matrix: {matrix}")

# Flatten a nested list
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested_list for item in sublist]
print(f"Flattened: {flattened}")

# 2. Multiple Conditions and Complex Logic
print("\n2. Multiple Conditions and Complex Logic:")
numbers = list(range(1, 21))

# Numbers divisible by both 2 and 3
divisible_by_2_and_3 = [x for x in numbers if x % 2 == 0 and x % 3 == 0]
print(f"Divisible by 2 and 3: {divisible_by_2_and_3}")

# Numbers that are either prime or perfect squares
perfect_squares = [x for x in range(1, 21) if int(x**0.5)**2 == x]
print(f"Perfect squares: {perfect_squares}")

# 3. Multiple Iterables with zip()
print("\n3. Multiple Iterables with zip():")
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["NYC", "LA", "Chicago"]

# Combine multiple lists
people_info = [f"{name} ({age}) from {city}" for name, age, city in zip(names, ages, cities)]
print(f"People info: {people_info}")

# 4. Dictionary Comprehension from Lists
print("\n4. Dictionary Comprehension from Lists:")
fruits = ["apple", "banana", "orange", "grape"]
prices = [1.50, 0.75, 2.00, 3.50]

# Create dictionary from two lists
fruit_prices = {fruit: price for fruit, price in zip(fruits, prices)}
print(f"Fruit prices: {fruit_prices}")

# 5. Complex Transformations
print("\n5. Complex Transformations:")
sentences = ["hello world", "python programming", "list comprehension"]

# Capitalize first letter of each word
capitalized = [" ".join(word.capitalize() for word in sentence.split()) for sentence in sentences]
print(f"Capitalized: {capitalized}")

# Count vowels in each sentence
vowel_counts = [sum(1 for char in sentence.lower() if char in 'aeiou') for sentence in sentences]
print(f"Vowel counts: {vowel_counts}")

# 6. Conditional Expressions (Ternary Operator)
print("\n6. Conditional Expressions:")
numbers = list(range(-5, 6))

# Convert negative to positive, keep positive as is
absolute_values = [x if x >= 0 else -x for x in numbers]
print(f"Absolute values: {absolute_values}")

# Categorize numbers
categories = ["positive" if x > 0 else "negative" if x < 0 else "zero" for x in numbers]
print(f"Categories: {categories}")

# 7. List Comprehension with Functions
print("\n7. List Comprehension with Functions:")
import math

# Calculate factorial for numbers 1-5
factorials = [math.factorial(x) for x in range(1, 6)]
print(f"Factorials: {factorials}")

# Check if numbers are prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [x for x in range(2, 21) if is_prime(x)]
print(f"Primes: {primes}")

# 8. Advanced Filtering and Mapping
print("\n8. Advanced Filtering and Mapping:")
students = [
    {"name": "Alice", "grades": [85, 90, 92]},
    {"name": "Bob", "grades": [78, 85, 80]},
    {"name": "Charlie", "grades": [92, 88, 95]},
    {"name": "Diana", "grades": [75, 82, 79]}
]

# Students with average grade > 85
high_achievers = [student["name"] for student in students 
                  if sum(student["grades"]) / len(student["grades"]) > 85]
print(f"High achievers: {high_achievers}")

# Average grades for each student
avg_grades = [sum(student["grades"]) / len(student["grades"]) for student in students]
print(f"Average grades: {avg_grades}")

# 9. List Comprehension with enumerate()
print("\n9. List Comprehension with enumerate():")
words = ["python", "programming", "list", "comprehension"]

# Add index to words
indexed_words = [f"{i+1}. {word}" for i, word in enumerate(words)]
print(f"Indexed words: {indexed_words}")

# Find words with even indices
even_index_words = [word for i, word in enumerate(words) if i % 2 == 0]
print(f"Even index words: {even_index_words}")

# 10. Real-world Example: Data Processing
print("\n10. Real-world Example - Data Processing:")
sales_data = [
    {"product": "laptop", "sales": 1200, "region": "North"},
    {"product": "phone", "sales": 800, "region": "South"},
    {"product": "tablet", "sales": 600, "region": "North"},
    {"product": "laptop", "sales": 900, "region": "South"},
    {"product": "phone", "sales": 1100, "region": "North"}
]

# Products with sales > 1000
high_sales_products = [item["product"] for item in sales_data if item["sales"] > 1000]
print(f"High sales products: {high_sales_products}")

# Total sales by region
north_sales = sum(item["sales"] for item in sales_data if item["region"] == "North")
south_sales = sum(item["sales"] for item in sales_data if item["region"] == "South")
print(f"North sales: ${north_sales}, South sales: ${south_sales}")

# 11. Performance Comparison
print("\n11. Performance Comparison:")
import time

# Traditional for loop
start_time = time.time()
traditional_result = []
for i in range(1000000):
    if i % 2 == 0:
        traditional_result.append(i**2)
traditional_time = time.time() - start_time

# List comprehension
start_time = time.time()
comprehension_result = [i**2 for i in range(1000000) if i % 2 == 0]
comprehension_time = time.time() - start_time

print(f"Traditional loop time: {traditional_time:.4f} seconds")
print(f"List comprehension time: {comprehension_time:.4f} seconds")
print(f"Comprehension is {traditional_time/comprehension_time:.1f}x faster")

# 12. Advanced Pattern: Cartesian Product
print("\n12. Advanced Pattern - Cartesian Product:")
colors = ["red", "blue", "green"]
sizes = ["S", "M", "L"]

# All possible color-size combinations
combinations = [f"{color} {size}" for color in colors for size in sizes]
print(f"Color-size combinations: {combinations}")

# 13. List Comprehension with Sets
print("\n13. List Comprehension with Sets:")
text = "hello world python programming"

# Unique characters (case-insensitive)
unique_chars = list({char.lower() for char in text if char.isalpha()})
print(f"Unique characters: {unique_chars}")

# 14. Error Handling in List Comprehension
print("\n14. Error Handling in List Comprehension:")
mixed_data = ["123", "456", "abc", "789", "def"]

# Safe conversion to integers
safe_numbers = []
for item in mixed_data:
    try:
        safe_numbers.append(int(item))
    except ValueError:
        pass

# Alternative with list comprehension (using a helper function)
def safe_int(x):
    try:
        return int(x)
    except ValueError:
        return None

safe_numbers_comp = [safe_int(item) for item in mixed_data if safe_int(item) is not None]
print(f"Safe numbers: {safe_numbers_comp}")

# ===== NESTED LIST COMPREHENSION DETAILED EXPLANATION =====
print("\n=== NESTED LIST COMPREHENSION DETAILED EXPLANATION ===")

# 1. Basic Nested List Comprehension Structure
print("\n1. Basic Structure:")
print("Syntax: [expression for outer_item in outer_iterable for inner_item in inner_iterable]")
print("This is equivalent to:")
print("result = []")
print("for outer_item in outer_iterable:")
print("    for inner_item in inner_iterable:")
print("        result.append(expression)")

# 2. Simple Example: Creating Pairs
print("\n2. Creating Pairs:")
# Traditional nested loops
pairs_traditional = []
for i in range(1, 4):
    for j in range(1, 4):
        pairs_traditional.append((i, j))

# Nested list comprehension
pairs_comprehension = [(i, j) for i in range(1, 4) for j in range(1, 4)]
print(f"Traditional: {pairs_traditional}")
print(f"Comprehension: {pairs_comprehension}")

# 3. Matrix Creation
print("\n3. Matrix Creation:")
# Create a 3x3 matrix with values i+j
matrix_3x3 = [[i+j for j in range(3)] for i in range(0, 9, 3)]
print(f"3x3 Matrix: {matrix_3x3}")

# Create a multiplication table
multiplication_table = [[i*j for j in range(1, 6)] for i in range(1, 6)]
print(f"5x5 Multiplication Table:")
for row in multiplication_table:
    print(f"  {row}")

# 4. Flattening Nested Lists
print("\n4. Flattening Nested Lists:")
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Traditional flattening
flattened_traditional = []
for sublist in nested_list:
    for item in sublist:
        flattened_traditional.append(item)

# Nested comprehension flattening
flattened_comprehension = [item for sublist in nested_list for item in sublist]
print(f"Original: {nested_list}")
print(f"Flattened: {flattened_comprehension}")

# 5. Conditional Nested Comprehension
print("\n5. Conditional Nested Comprehension:")
# Only include even numbers from nested lists
nested_with_odds = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
even_only = [item for sublist in nested_with_odds for item in sublist if item % 2 == 0]
print(f"Original: {nested_with_odds}")
print(f"Even numbers only: {even_only}")

# 6. Complex Nested Comprehension with Multiple Conditions
print("\n6. Complex Nested Comprehension:")
# Create a list of coordinates where both x and y are even
coordinates = [(x, y) for x in range(1, 6) for y in range(1, 6) if x % 2 == 0 and y % 2 == 0]
print(f"Coordinates with even x and y: {coordinates}")

# 7. Nested Comprehension with Different Data Types
print("\n7. Different Data Types:")
# Create a list of strings from nested lists of numbers
nested_numbers = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
number_strings = [str(num) for sublist in nested_numbers for num in sublist]
print(f"Original: {nested_numbers}")
print(f"As strings: {number_strings}")

# 8. Nested Comprehension with Functions
print("\n8. Nested Comprehension with Functions:")
def square(x):
    return x ** 2

def is_even(x):
    return x % 2 == 0

# Apply functions in nested comprehension
nested_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
squared_evens = [square(num) for sublist in nested_data for num in sublist if is_even(num)]
print(f"Original: {nested_data}")
print(f"Squared even numbers: {squared_evens}")

# 9. Multiple Levels of Nesting
print("\n9. Multiple Levels of Nesting:")
# 3D structure: list of lists of lists
deeply_nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
print(f"Deeply nested: {deeply_nested}")

# Flatten 3D structure
flattened_3d = [item for sublist1 in deeply_nested 
                for sublist2 in sublist1 
                for item in sublist2]
print(f"Flattened 3D: {flattened_3d}")

# 10. Nested Comprehension with enumerate()
print("\n10. Nested Comprehension with enumerate():")
words_list = [["hello", "world"], ["python", "programming"], ["list", "comprehension"]]

# Add indices to each word
indexed_words = [f"{outer_idx}-{inner_idx}: {word}" 
                for outer_idx, sublist in enumerate(words_list)
                for inner_idx, word in enumerate(sublist)]
print(f"Original: {words_list}")
print(f"Indexed: {indexed_words}")

# 11. Real-world Example: Student Grades
print("\n11. Real-world Example - Student Grades:")
students_grades = [
    {"name": "Alice", "grades": [85, 90, 92]},
    {"name": "Bob", "grades": [78, 85, 80]},
    {"name": "Charlie", "grades": [92, 88, 95]}
]

# Get all grades above 85
high_grades = [grade for student in students_grades 
               for grade in student["grades"] 
               if grade > 85]
print(f"All grades above 85: {high_grades}")

low_grades = [grade for student in students_grades 
      for grade in student["grades"] if grade < 90]
print(f"All grades below 90: {low_grades}")



# Get student names with their high grades
high_grade_students = [(student["name"], grade) for student in students_grades 
                       for grade in student["grades"] 
                       if grade > 85]
print(f"Students with high grades: {high_grade_students}")

# 12. Performance Comparison: Nested vs Traditional
print("\n12. Performance Comparison:")
import time

# Traditional nested loops
start_time = time.time()
traditional_result = []
for i in range(100):
    for j in range(100):
        if i % 2 == 0 and j % 2 == 0:
            traditional_result.append(i * j)
traditional_time = time.time() - start_time

# Nested comprehension
start_time = time.time()
comprehension_result = [i * j for i in range(100) for j in range(100) if i % 2 == 0 and j % 2 == 0]
comprehension_time = time.time() - start_time

print(f"Traditional nested loops: {traditional_time:.4f} seconds")
print(f"Nested comprehension: {comprehension_time:.4f} seconds")
print(f"Comprehension is {traditional_time/comprehension_time:.1f}x faster")

# 13. Common Patterns and Tips
print("\n13. Common Patterns and Tips:")
print("Pattern 1: Flattening")
print("  [item for sublist in nested_list for item in sublist]")
print()
print("Pattern 2: Matrix creation")
print("  [[expression for j in range(cols)] for i in range(rows)]")
print()
print("Pattern 3: Conditional flattening")
print("  [item for sublist in nested_list for item in sublist if condition]")
print()
print("Pattern 4: Multiple conditions")
print("  [expression for outer in outer_iter for inner in inner_iter if outer_condition and inner_condition]")

# 14. Debugging Nested Comprehensions
print("\n14. Debugging Nested Comprehensions:")
print("Tip: Break down complex nested comprehensions into steps:")
print("Step 1: Write the traditional nested loops")
print("Step 2: Convert to comprehension step by step")
print("Step 3: Add conditions and transformations")

# Example of step-by-step conversion
print("\nExample - Step by step conversion:")
print("Traditional:")
print("result = []")
print("for i in range(3):")
print("    for j in range(3):")
print("        if i != j:")
print("            result.append((i, j))")

print("\nComprehension:")
print("result = [(i, j) for i in range(3) for j in range(3) if i != j]")

# Demonstrate the result
result = [(i, j) for i in range(3) for j in range(3) if i != j]
print(f"Result: {result}")

# ===== ADVANCED LAMBDA & FUNCTIONAL PROGRAMMING EXAMPLES =====
print("\n=== ADVANCED LAMBDA & FUNCTIONAL PROGRAMMING EXAMPLES ===")

# 1. Basic Lambda Functions
print("\n1. Basic Lambda Functions:")
# Simple arithmetic operations
add = lambda x, y: x + y
multiply = lambda x, y: x * y
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0

print(f"add(5, 3) = {add(5, 3)}")
print(f"multiply(4, 7) = {multiply(4, 7)}")
print(f"square(6) = {square(6)}")
print(f"is_even(8) = {is_even(8)}")

# 2. Lambda with map() - Functional Programming
print("\n2. Lambda with map():")
numbers = [1, 2, 3, 4, 5]

# Square all numbers
squared = list(map(lambda x: x**2, numbers))
print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Convert to strings
string_numbers = list(map(lambda x: str(x), numbers))
print(f"As strings: {string_numbers}")

# 3. Lambda with filter() - Functional Programming
print("\n3. Lambda with filter():")
# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {evens}")

# Filter numbers greater than 3
greater_than_3 = list(filter(lambda x: x > 3, numbers))
print(f"Greater than 3: {greater_than_3}")

# 4. Lambda with reduce() - Functional Programming
print("\n4. Lambda with reduce():")
from functools import reduce

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(f"Sum: {total}")

# Find maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(f"Maximum: {maximum}")

# Multiply all numbers
product = reduce(lambda x, y: x * y, numbers)
print(f"Product: {product}")

# 5. Lambda with sorted() - Custom Sorting
print("\n5. Lambda with sorted():")
students = [
    {"name": "Alice", "age": 20, "grade": 85},
    {"name": "Bob", "age": 22, "grade": 92},
    {"name": "Charlie", "age": 19, "grade": 78},
    {"name": "Diana", "age": 21, "grade": 88}
]

# Sort by age
sorted_by_age = sorted(students, key=lambda x: x["age"])
print(f"Sorted by age: {[s['name'] for s in sorted_by_age]}")

# Sort by grade (descending)
sorted_by_grade = sorted(students, key=lambda x: x["grade"], reverse=True)
print(f"Sorted by grade (descending): {[s['name'] for s in sorted_by_grade]}")

# 6. Lambda with Multiple Conditions
print("\n6. Lambda with Multiple Conditions:")
# Complex filtering
complex_filter = list(filter(lambda x: x % 2 == 0 and x > 3, numbers))
print(f"Even numbers > 3: {complex_filter}")

# Complex sorting
words = ["python", "programming", "lambda", "functional", "code"]
sorted_by_length_then_alpha = sorted(words, key=lambda x: (len(x), x))
print(f"Sorted by length then alphabetically: {sorted_by_length_then_alpha}")

# 7. Higher-Order Functions with Lambda
print("\n7. Higher-Order Functions:")
def apply_operation(func, x, y):
    return func(x, y)

# Using lambda with higher-order function
result1 = apply_operation(lambda a, b: a + b, 10, 5)
result2 = apply_operation(lambda a, b: a * b, 10, 5)
result3 = apply_operation(lambda a, b: a ** b, 2, 3)

print(f"Addition: {result1}")
print(f"Multiplication: {result2}")
print(f"Exponentiation: {result3}")

# 8. Lambda with List Comprehensions
print("\n8. Lambda with List Comprehensions:")
# Create list of lambda functions
operations = [lambda x, i=i: x + i for i in range(5)]
print(f"Operations: {[op(10) for op in operations]}")

# Lambda in comprehension
squares_and_cubes = [(lambda x: x**2)(i), (lambda x: x**3)(i)] for i in range(1, 4)]
print(f"Squares and cubes: {squares_and_cubes}")

# 9. Lambda with Partial Functions
print("\n9. Lambda with Partial Functions:")
from functools import partial

# Create partial functions with lambda
add_five = partial(lambda x, y: x + y, 5)
multiply_by_three = partial(lambda x, y: x * y, 3)

print(f"add_five(10) = {add_five(10)}")
print(f"multiply_by_three(7) = {multiply_by_three(7)}")

# 10. Lambda with Error Handling
print("\n10. Lambda with Error Handling:")
def safe_divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "Error: Division by zero"

safe_divide_lambda = lambda x, y: x / y if y != 0 else "Error: Division by zero"

print(f"safe_divide_lambda(10, 2) = {safe_divide_lambda(10, 2)}")
print(f"safe_divide_lambda(10, 0) = {safe_divide_lambda(10, 0)}")

# 11. Lambda with Closures
print("\n11. Lambda with Closures:")
def create_multiplier(factor):
    return lambda x: x * factor

double = create_multiplier(2)
triple = create_multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")

# 12. Advanced Functional Programming Patterns
print("\n12. Advanced Functional Programming Patterns:")
# Pipeline processing
def pipeline(*functions):
    return lambda x: reduce(lambda result, func: func(result), functions, x)

# Example pipeline: square -> add 10 -> multiply by 2
pipeline_func = pipeline(
    lambda x: x**2,
    lambda x: x + 10,
    lambda x: x * 2
)

print(f"Pipeline result for 5: {pipeline_func(5)}")

# 13. Lambda with Data Processing
print("\n13. Lambda with Data Processing:")
sales_data = [
    {"product": "laptop", "sales": 1200, "region": "North"},
    {"product": "phone", "sales": 800, "region": "South"},
    {"product": "tablet", "sales": 600, "region": "North"},
    {"product": "laptop", "sales": 900, "region": "South"}
]

# Extract and process data
product_names = list(map(lambda x: x["product"], sales_data))
high_sales = list(filter(lambda x: x["sales"] > 1000, sales_data))
total_sales = reduce(lambda acc, x: acc + x["sales"], sales_data, 0)

print(f"Product names: {product_names}")
print(f"High sales items: {[item['product'] for item in high_sales]}")
print(f"Total sales: ${total_sales}")

# 14. Lambda with Mathematical Operations
print("\n14. Lambda with Mathematical Operations:")
import math

# Mathematical functions as lambdas
factorial = lambda n: reduce(lambda x, y: x * y, range(1, n + 1), 1)
fibonacci = lambda n: reduce(lambda x, _: [x[1], x[0] + x[1]], range(n), [0, 1])[0]
is_prime = lambda n: n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))

print(f"factorial(5) = {factorial(5)}")
print(f"fibonacci(8) = {fibonacci(8)}")
print(f"is_prime(17) = {is_prime(17)}")

# 15. Lambda with String Processing
print("\n15. Lambda with String Processing:")
text = "Hello World Python Programming"

# String processing with lambda
word_lengths = list(map(lambda word: len(word), text.split()))
long_words = list(filter(lambda word: len(word) > 5, text.split()))
reversed_words = list(map(lambda word: word[::-1], text.split()))

print(f"Word lengths: {word_lengths}")
print(f"Long words: {long_words}")
print(f"Reversed words: {reversed_words}")

# 16. Performance Comparison: Lambda vs Regular Functions
print("\n16. Performance Comparison:")
import time

# Regular function
def square_func(x):
    return x ** 2

# Lambda function
square_lambda = lambda x: x ** 2

# Test performance
numbers_test = list(range(1000000))

# Regular function timing
start_time = time.time()
result_func = list(map(square_func, numbers_test))
func_time = time.time() - start_time

# Lambda function timing
start_time = time.time()
result_lambda = list(map(square_lambda, numbers_test))
lambda_time = time.time() - start_time

print(f"Regular function time: {func_time:.4f} seconds")
print(f"Lambda function time: {lambda_time:.4f} seconds")
print(f"Lambda is {func_time/lambda_time:.1f}x faster")

# 17. Real-world Example: Data Analysis Pipeline
print("\n17. Real-world Example - Data Analysis Pipeline:")
# Simulate data processing pipeline
raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Pipeline: filter -> transform -> aggregate
processed_data = reduce(
    lambda acc, x: acc + x,
    map(
        lambda x: x ** 2,
        filter(lambda x: x % 2 == 0, raw_data)
    ),
    0
)

print(f"Raw data: {raw_data}")
print(f"Pipeline result (sum of squared evens): {processed_data}")

# 18. Lambda with Decorators
print("\n18. Lambda with Decorators:")
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Using lambda with decorator
@timer
def process_data(data):
    return list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, data)))

result = process_data(range(10000))
print(f"Processed {len(result)} items")




