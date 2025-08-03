# import squaremodule
from squaremodule import square



# print(squaremodule.square(4))

numbers = [1,5,8,9]

squared = list(map(square,numbers ))

print(squared)

print(square(4))

# numbers = [1,5,8,9]

# squared = list(map(square,numbers ))

# print(squared)

# ===== ADVANCED LAMBDA & FUNCTIONAL PROGRAMMING EXAMPLES =====
print("\n=== ADVANCED LAMBDA & FUNCTIONAL PROGRAMMING EXAMPLES ===")

# 1. Basic Lambda Functions
print("\n1. Basic Lambda Functions:")
# Simple arithmetic operations
add = lambda x, y: x + y
multiply = lambda x, y: x * y
square_lambda = lambda x: x ** 2
is_even = lambda x: x % 2 == 0

print(f"add(5, 3) = {add(5, 3)}")
print(f"multiply(4, 7) = {multiply(4, 7)}")
print(f"square_lambda(6) = {square_lambda(6)}")
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
square_lambda_func = lambda x: x ** 2

# Test performance
numbers_test = list(range(1000000))

# Regular function timing
start_time = time.time()
result_func = list(map(square_func, numbers_test))
func_time = time.time() - start_time

# Lambda function timing
start_time = time.time()
result_lambda = list(map(square_lambda_func, numbers_test))
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

# 19. Lambda with Nested Functions
print("\n19. Lambda with Nested Functions:")
def create_calculator():
    def add(x, y):
        return lambda: x + y
    
    def multiply(x, y):
        return lambda: x * y
    
    return add, multiply

add_func, multiply_func = create_calculator()
print(f"Add result: {add_func(5, 3)()}")
print(f"Multiply result: {multiply_func(4, 7)()}")

# 20. Lambda with Generators
print("\n20. Lambda with Generators:")
def number_generator():
    for i in range(1, 6):
        yield i

# Process generator with lambda
squared_generator = map(lambda x: x**2, number_generator())
print(f"Squared generator values: {list(squared_generator)}")

# 21. Lambda with Classes
print("\n21. Lambda with Classes:")
class Calculator:
    def __init__(self):
        self.operations = {
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'multiply': lambda x, y: x * y,
            'divide': lambda x, y: x / y if y != 0 else "Error"
        }
    
    def calculate(self, operation, x, y):
        return self.operations[operation](x, y)

calc = Calculator()
print(f"Calculator add: {calc.calculate('add', 10, 5)}")
print(f"Calculator multiply: {calc.calculate('multiply', 4, 7)}")

# 22. Lambda with Exception Handling
print("\n22. Lambda with Exception Handling:")
def safe_operation(operation):
    def wrapper(*args):
        try:
            return operation(*args)
        except Exception as e:
            return f"Error: {e}"
    return wrapper

safe_sqrt = safe_operation(lambda x: x ** 0.5)
print(f"Safe sqrt(4): {safe_sqrt(4)}")
print(f"Safe sqrt(-1): {safe_sqrt(-1)}")

# 23. Lambda with Recursion
print("\n23. Lambda with Recursion:")
# Factorial using lambda and recursion
factorial_recursive = lambda n: 1 if n <= 1 else n * factorial_recursive(n - 1)
print(f"Factorial recursive(5): {factorial_recursive(5)}")

# 24. Lambda with Context Managers
print("\n24. Lambda with Context Managers:")
from contextlib import contextmanager

@contextmanager
def timer_context():
    start = time.time()
    yield lambda: time.time() - start

with timer_context() as get_time:
    # Do some work
    result = list(map(lambda x: x**2, range(1000)))
    elapsed = get_time()
    print(f"Work completed in {elapsed:.4f} seconds")

# 25. Lambda with Async Programming (Conceptual)
print("\n25. Lambda with Async Programming (Conceptual):")
import asyncio

async def async_operation():
    # Simulate async work
    await asyncio.sleep(0.1)
    return "Async result"

# Lambda that returns async function
async_lambda = lambda: async_operation()

# Note: This is conceptual - actual async lambdas have limitations
print("Async lambda concept demonstrated")

# 26. Lambda with Type Hints (Python 3.9+)
print("\n26. Lambda with Type Hints:")
from typing import Callable, List, Union

# Type hints for lambda functions
number_operation: Callable[[int, int], int] = lambda x, y: x + y
string_operation: Callable[[str, str], str] = lambda x, y: x + " " + y

print(f"Number operation: {number_operation(5, 3)}")
print(f"String operation: {string_operation('Hello', 'World')}")

# 27. Lambda with Memory Management
print("\n27. Lambda with Memory Management:")
import gc

# Create many lambda functions
lambda_functions = [lambda x, i=i: x + i for i in range(1000)]

# Use them
results = [func(10) for func in lambda_functions[:5]]
print(f"First 5 results: {results}")

# Clean up
del lambda_functions
gc.collect()
print("Memory cleaned up")

# 28. Lambda with Caching
print("\n28. Lambda with Caching:")
from functools import lru_cache

# Cached lambda (wrapped in function)
@lru_cache(maxsize=128)
def cached_square(x):
    return lambda y: x ** y

square_2 = cached_square(2)
print(f"Cached square_2(5): {square_2(5)}")
print(f"Cached square_2(5) again: {square_2(5)}")  # Should be faster

# 29. Lambda with Metaclasses (Advanced)
print("\n29. Lambda with Metaclasses (Advanced):")
class LambdaMeta(type):
    def __new__(cls, name, bases, attrs):
        # Add a lambda method to all classes using this metaclass
        attrs['lambda_method'] = lambda self, x: x * 2
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=LambdaMeta):
    pass

obj = MyClass()
print(f"Lambda method result: {obj.lambda_method(5)}")

# 30. Lambda with Threading
print("\n30. Lambda with Threading:")
import threading

def threaded_operation(operation, data, results, index):
    results[index] = operation(data)

# Use lambda with threading
data_list = [1, 2, 3, 4, 5]
results = [None] * len(data_list)
threads = []

for i, data in enumerate(data_list):
    thread = threading.Thread(
        target=threaded_operation,
        args=(lambda x: x**2, data, results, i)
    )
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"Threaded results: {results}")

