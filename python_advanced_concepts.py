# Python Advanced Concepts - Quick Revision for Full-Stack Development
# Perfect for JS/TS developers jumping straight into Python backend

from typing import List, Dict, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from contextlib import contextmanager
import asyncio
import json
from datetime import datetime
from enum import Enum

# ============================================================================
# 1. TYPE HINTS & ANNOTATIONS (Like TypeScript)
# ============================================================================

# Basic type hints
def get_user_name(user_id: int) -> str:
    return f"User {user_id}"

# Complex types
def process_users(users: List[Dict[str, Any]]) -> Optional[List[str]]:
    if not users:
        return None
    return [user.get("name", "") for user in users]

# Union types (like TypeScript union)
def handle_id(user_id: Union[int, str]) -> str:
    return str(user_id)

# Generic types
from typing import TypeVar, Generic
T = TypeVar('T')

class ApiResponse(Generic[T]):
    def __init__(self, data: T, status: int = 200):
        self.data = data
        self.status = status

# ============================================================================
# 2. DATACLASSES (Like TypeScript interfaces/classes)
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, date

# Basic dataclass
@dataclass
class Person:
    name: str
    age: int
    email: str

# Dataclass with default values
@dataclass
class Product:
    name: str
    price: float
    category: str = "General"
    in_stock: bool = True

# Dataclass with field() for complex defaults
@dataclass
class User:
    username: str
    email: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

# Dataclass with post-init processing
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not included in __init__
    
    def __post_init__(self):
        self.area = self.width * self.height

# Frozen dataclass (immutable)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# Dataclass with ordering
@dataclass(order=True)
class Student:
    name: str
    grade: float
    age: int = field(compare=False)  # Exclude from comparison

# More complex example with nested dataclasses
@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str

@dataclass
class Employee:
    name: str
    employee_id: int
    department: str
    salary: float
    address: Optional[Address] = None
    skills: List[str] = field(default_factory=list)
    hire_date: date = field(default_factory=date.today)

# Example usage
if __name__ == "__main__":
    print("=== Basic Person ===")
    person1 = Person("Alice", 30, "alice@example.com")
    print(f"Person: {person1}")
    print(f"Name: {person1.name}, Age: {person1.age}")
    
    print("\n=== Product with defaults ===")
    product1 = Product("Laptop", 999.99)
    product2 = Product("Mouse", 29.99, "Electronics", False)
    print(f"Product 1: {product1}")
    print(f"Product 2: {product2}")
    
    print("\n=== User with field() defaults ===")
    user1 = User("john_doe", "john@example.com")
    user2 = User("jane_smith", "jane@example.com", ["admin", "editor"])
    print(f"User 1: {user1}")
    print(f"User 2: {user2}")
    
    print("\n=== Rectangle with post-init ===")
    rect = Rectangle(5.0, 3.0)
    print(f"Rectangle: {rect}")
    print(f"Area calculated: {rect.area}")
    
    print("\n=== Frozen Point (immutable) ===")
    point1 = Point(1.0, 2.0)
    point2 = Point(1.0, 2.0)
    print(f"Point 1: {point1}")
    print(f"Points equal: {point1 == point2}")
    # point1.x = 3.0  # This would raise an error!
    
    print("\n=== Ordered Students ===")
    student1 = Student("Alice", 85.5, 20)
    student2 = Student("Bob", 92.0, 19)
    student3 = Student("Charlie", 78.5, 21)
    students = [student1, student2, student3]
    
    print("Students before sorting:")
    for s in students:
        print(f"  {s}")
    
    students.sort()
    print("\nStudents after sorting by grade:")
    for s in students:
        print(f"  {s}")
    
    print("\n=== Complex Employee example ===")
    address = Address("123 Main St", "New York", "USA", "10001")
    employee = Employee(
        name="Sarah Johnson",
        employee_id=1001,
        department="Engineering",
        salary=75000.0,
        address=address,
        skills=["Python", "JavaScript", "SQL"]
    )
    print(f"Employee: {employee}")
    print(f"Lives in: {employee.address.city}, {employee.address.country}")
    print(f"Skills: {', '.join(employee.skills)}")
    
    print("\n=== Dataclass methods ===")
    # Dataclasses automatically generate __repr__, __eq__, __hash__ (if frozen)
    person2 = Person("Alice", 30, "alice@example.com")
    print(f"person1 == person2: {person1 == person2}")
    print(f"person1 is person2: {person1 is person2}")
    
    # Converting to dict (not built-in, but useful)
    from dataclasses import asdict, astuple
    print(f"\nPerson as dict: {asdict(person1)}")
    print(f"Person as tuple: {astuple(person1)}")
    
    # Fields inspection
    from dataclasses import fields
    print(f"\nPerson fields: {[f.name for f in fields(Person)]}")
    
    print("\n=== Modifying mutable defaults safely ===")
    user1.tags.append("premium")
    user2.tags.append("vip")
    print(f"User 1 tags: {user1.tags}")
    print(f"User 2 tags: {user2.tags}")  # Each instance has its own list!

# ============================================================================
# 3. DECORATORS (Like TypeScript decorators)
# ============================================================================

# Basic decorator
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f"{func.__name__} took {end - start}")
        return result
    return wrapper

# Decorator with parameters
def validate_user_role(required_role: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In real app, get user from JWT token
            user_role = kwargs.get('user_role', 'user')
            if user_role != required_role:
                raise PermissionError(f"Required role: {required_role}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Class decorator
def api_endpoint(path: str):
    def decorator(cls):
        cls.api_path = path
        return cls
    return decorator

# Usage examples
@log_execution_time
def slow_operation():
    import time
    time.sleep(1)
    return "Done"

@validate_user_role("admin")
def delete_user(user_id: int, user_role: str = "user"):
    return f"Deleted user {user_id}"

# ============================================================================
# 4. CONTEXT MANAGERS (Like try/finally but cleaner)
# ============================================================================

# Custom context manager
@contextmanager
def database_transaction():
    print("Starting transaction")
    try:
        yield "db_connection"
        print("Committing transaction")
    except Exception as e:
        print(f"Rolling back transaction: {e}")
        raise
    finally:
        print("Closing connection")

# Usage
def create_user_with_transaction(user_data: Dict[str, Any]):
    with database_transaction() as db:
        # Simulate database operation
        print(f"Creating user with {db}: {user_data}")
        return User(id=1, name=user_data["name"], email=user_data["email"])

# File handling context manager
def read_config_file(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as file:
        return json.load(file)

# ============================================================================
# 4.1 ADVANCED CONTEXT MANAGERS FOR FULL-STACK DEVELOPMENT
# ============================================================================

import time
import threading
from typing import Generator, Optional
import requests
from contextlib import asynccontextmanager

# 1. Database Connection Pool Context Manager
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
        self.connection = f"DB_CONNECTION_{id(self)}"  # Simulate connection
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing database connection: {self.connection}")
        self.connection = None
        if exc_type:
            print(f"Database error occurred: {exc_val}")
            return False  # Re-raise the exception
        return True

# 2. API Rate Limiting Context Manager
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def __enter__(self):
        with self.lock:
            current_time = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (current_time - self.requests[0])
                if sleep_time > 0:
                    print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
            
            self.requests.append(current_time)
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No cleanup needed

# 3. Cache Context Manager
class CacheContext:
    def __init__(self, cache_key: str, ttl: int = 300):
        self.cache_key = cache_key
        self.ttl = ttl
        self.cache = {}  # In real app, this would be Redis/Memcached
    
    def __enter__(self):
        if self.cache_key in self.cache:
            cached_data, timestamp = self.cache[self.cache_key]
            if time.time() - timestamp < self.ttl:
                print(f"Cache hit for key: {self.cache_key}")
                return cached_data
        print(f"Cache miss for key: {self.cache_key}")
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # No exception occurred
            # In real implementation, you'd store the result here
            pass

# 4. Logging Context Manager
@contextmanager
def log_operation(operation_name: str, **kwargs):
    start_time = time.time()
    print(f"Starting {operation_name} with params: {kwargs}")
    try:
        yield
        print(f"Completed {operation_name} successfully in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Failed {operation_name} after {time.time() - start_time:.2f}s: {e}")
        raise
    finally:
        print(f"Finished {operation_name}")

# 5. Temporary File Context Manager
@contextmanager
def temporary_file(prefix: str = "temp", suffix: str = ".tmp"):
    import tempfile
    import os
    
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    try:
        yield temp_file.name
    finally:
        temp_file.close()
        os.unlink(temp_file.name)  # Delete the file

# 6. Async Context Manager for API Calls
@asynccontextmanager
async def api_session(base_url: str, timeout: int = 30):
    session = requests.Session()
    session.timeout = timeout
    try:
        yield session
    finally:
        session.close()

# 7. Database Transaction with Rollback
@contextmanager
def database_transaction_with_rollback(db_connection):
    transaction_id = f"TXN_{int(time.time())}"
    print(f"Starting transaction {transaction_id}")
    
    try:
        # Simulate database operations
        yield transaction_id
        print(f"Committing transaction {transaction_id}")
    except Exception as e:
        print(f"Rolling back transaction {transaction_id} due to: {e}")
        raise
    finally:
        print(f"Transaction {transaction_id} completed")

# 8. Resource Pool Context Manager
class ConnectionPool:
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.active_connections = 0
        self.lock = threading.Lock()
    
    def __enter__(self):
        with self.lock:
            if self.active_connections >= self.max_connections:
                raise RuntimeError("No available connections in pool")
            self.active_connections += 1
            connection_id = f"CONN_{self.active_connections}"
            print(f"Acquired connection {connection_id}")
            return connection_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.lock:
            self.active_connections -= 1
            print(f"Released connection. Active: {self.active_connections}")

# 9. Performance Monitoring Context Manager
@contextmanager
def performance_monitor(operation_name: str):
    start_time = time.time()
    start_memory = 0  # In real app, use psutil to get memory usage
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Performance: {operation_name} took {duration:.4f}s")

# 10. Configuration Context Manager
@contextmanager
def configuration_context(config_file: str):
    import os
    original_env = os.environ.copy()
    
    try:
        # Load configuration from file
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = str(value)
        
        yield config
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

# 11. File Lock Context Manager
@contextmanager
def file_lock(filename: str):
    lock_file = f"{filename}.lock"
    
    try:
        # Try to acquire lock
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        yield
    finally:
        # Release lock
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass

# 12. API Authentication Context Manager
@contextmanager
def authenticated_session(api_key: str, base_url: str):
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    })
    
    try:
        yield session
    finally:
        session.close()

# 13. Database Migration Context Manager
@contextmanager
def database_migration(version: str, rollback_on_error: bool = True):
    print(f"Starting migration to version {version}")
    backup_created = False
    
    try:
        # Create backup
        print("Creating database backup...")
        backup_created = True
        
        # Apply migration
        print(f"Applying migration {version}...")
        yield
        
        print(f"Migration {version} completed successfully")
    except Exception as e:
        print(f"Migration {version} failed: {e}")
        if rollback_on_error and backup_created:
            print("Rolling back to previous version...")
        raise
    finally:
        if backup_created:
            print("Cleaning up backup files...")

# 14. Memory Usage Context Manager
@contextmanager
def memory_monitor(threshold_mb: int = 100):
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        if memory_used > threshold_mb:
            print(f"Warning: High memory usage detected: {memory_used:.2f}MB")

# 15. Circuit Breaker Context Manager
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __enter__(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                print("Circuit breaker opened due to too many failures")
        else:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0

# Usage examples for advanced context managers
def demonstrate_advanced_context_managers():
    print("\n=== Advanced Context Managers Demo ===\n")
    
    # 1. Database connection
    print("1. Database Connection:")
    with DatabaseConnection("postgresql://localhost/mydb") as conn:
        print(f"Using connection: {conn}")
        # Simulate database operations
    
    # 2. Rate limiting
    print("\n2. Rate Limiting:")
    with RateLimiter(max_requests=3, time_window=1.0):
        print("API call 1")
    with RateLimiter(max_requests=3, time_window=1.0):
        print("API call 2")
    
    # 3. Logging operations
    print("\n3. Logging Operations:")
    with log_operation("user_creation", user_id=123, email="test@example.com"):
        time.sleep(0.1)  # Simulate work
    
    # 4. Performance monitoring
    print("\n4. Performance Monitoring:")
    with performance_monitor("data_processing"):
        time.sleep(0.05)  # Simulate processing
    
    # 5. Connection pool
    print("\n5. Connection Pool:")
    pool = ConnectionPool(max_connections=2)
    with pool as conn1:
        print(f"Using connection: {conn1}")
        with pool as conn2:
            print(f"Using connection: {conn2}")
    
    # 6. Circuit breaker
    print("\n6. Circuit Breaker:")
    breaker = CircuitBreaker(failure_threshold=2)
    try:
        with breaker:
            print("Operation 1 - should succeed")
    except Exception as e:
        print(f"Operation failed: {e}")
    
    # 7. Temporary file
    print("\n7. Temporary File:")
    with temporary_file(prefix="test_", suffix=".json") as temp_path:
        print(f"Created temporary file: {temp_path}")
        # File will be automatically deleted after this block
    
    # 8. Database transaction
    print("\n8. Database Transaction:")
    with database_transaction_with_rollback("mock_db") as txn_id:
        print(f"Performing operations in transaction {txn_id}")
        # Simulate successful transaction
    
    # 9. Cache context
    print("\n9. Cache Context:")
    with CacheContext("user_profile_123") as cached_data:
        if cached_data is None:
            print("Fetching data from database...")
            # Simulate database fetch
        else:
            print("Using cached data")

# ============================================================================
# 5. ASYNC/AWAIT (Very similar to JS async/await)
# ============================================================================

# Basic async function
async def fetch_user_async(user_id: int) -> User:
    # Simulate async database call
    await asyncio.sleep(0.1)
    return User(id=user_id, name=f"User {user_id}", email=f"user{user_id}@example.com")

# Async with multiple operations
async def get_user_with_posts(user_id: int) -> Dict[str, Any]:
    # Parallel execution (like Promise.all)
    user_task = fetch_user_async(user_id)
    posts_task = fetch_user_posts_async(user_id)
    
    user, posts = await asyncio.gather(user_task, posts_task)
    
    return {
        "user": user.to_dict(),
        "posts": posts
    }

async def fetch_user_posts_async(user_id: int) -> List[Dict[str, Any]]:
    await asyncio.sleep(0.1)
    return [
        {"id": 1, "title": "Post 1", "user_id": user_id},
        {"id": 2, "title": "Post 2", "user_id": user_id}
    ]

# Async generator
async def stream_users():
    for i in range(5):
        await asyncio.sleep(0.1)
        yield User(id=i, name=f"User {i}", email=f"user{i}@example.com")

# ============================================================================
# 6. GENERATORS & ITERATORS (Memory efficient)
# ============================================================================

# Generator function
def fibonacci_generator(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Generator expression (like array comprehension but lazy)
def get_even_squares(numbers: List[int]):
    return (x**2 for x in numbers if x % 2 == 0)

# Class-based iterator
class UserIterator:
    def __init__(self, users: List[User]):
        self.users = users
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.users):
            raise StopIteration
        user = self.users[self.index]
        self.index += 1
        return user

# ============================================================================
# 7. COMPREHENSIONS (Like array methods but more powerful)
# ============================================================================

# List comprehension (like Array.map + filter)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16, 36, 64, 100]

# Dictionary comprehension
user_names = ["Alice", "Bob", "Charlie"]
user_dict = {name: f"{name.lower()}@example.com" for name in user_names}

# Set comprehension
unique_lengths = {len(name) for name in user_names}

# Nested comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for row in matrix for item in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# ============================================================================
# 8. CLOSURES & HIGHER-ORDER FUNCTIONS
# ============================================================================

# Closure (like JavaScript closures)
def create_multiplier(factor: int) -> Callable[[int], int]:
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

# Higher-order function
def apply_operation(numbers: List[int], operation: Callable[[int], int]) -> List[int]:
    return [operation(x) for x in numbers]

# Partial application
from functools import partial

def add_numbers(a: int, b: int, c: int) -> int:
    return a + b + c

add_five_and_ten = partial(add_numbers, 5, 10)  # Now only needs one argument

# ============================================================================
# 9. METACLASSES & DESCRIPTORS (Advanced)
# ============================================================================

# Property descriptor
class ValidatedProperty:
    def __init__(self, validator: Callable[[Any], bool]):
        self.validator = validator
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)
    
    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}")
        setattr(instance, self.name, value)

# Usage
class ValidatedUser:
    email = ValidatedProperty(lambda x: "@" in x)
    age = ValidatedProperty(lambda x: isinstance(x, int) and x >= 0)
    
    def __init__(self, email: str, age: int):
        self.email = email
        self.age = age

# ============================================================================
# 10. ERROR HANDLING & CUSTOM EXCEPTIONS
# ============================================================================

# Custom exception hierarchy
class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ValidationError(APIError):
    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error for {field}: {message}", 422)

class NotFoundError(APIError):
    def __init__(self, resource: str, resource_id: Any):
        super().__init__(f"{resource} with id {resource_id} not found", 404)

# Error handling with context
def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        raise ValidationError("divisor", "Cannot divide by zero")
    except TypeError as e:
        raise ValidationError("operands", f"Invalid operand types: {e}")

# ============================================================================
# 11. CACHING & MEMOIZATION
# ============================================================================

# LRU Cache (like memoization)
@lru_cache(maxsize=128)
def expensive_calculation(n: int) -> int:
    print(f"Computing for {n}")  # This will only print once per unique n
    return n * n * n

# Custom cache decorator
def simple_cache(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

@simple_cache
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# ============================================================================
# 12. ENUMS & CONSTANTS
# ============================================================================

class UserRole(Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"

class HttpStatus(Enum):
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500

# ============================================================================
# 13. PRACTICAL EXAMPLES FOR FASTAPI BACKEND
# ============================================================================

# FastAPI-style dependency injection pattern
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def get_user(self, user_id: int) -> Optional[User]:
        # Simulate database query
        await asyncio.sleep(0.1)
        return User(id=user_id, name=f"User {user_id}", email=f"user{user_id}@example.com")

class UserService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
    
    async def create_user(self, user_request: UserCreateRequest) -> UserResponse:
        # Validation
        if not user_request.email or "@" not in user_request.email:
            raise ValidationError("email", "Invalid email format")
        
        # Create user
        user = User(
            id=1,  # In real app, this would be generated
            name=user_request.name,
            email=user_request.email
        )
        
        # Generate token (simplified)
        token = f"jwt_token_for_user_{user.id}"
        
        return UserResponse(user=user, token=token)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    print("=== Python Advanced Concepts Demo ===\n")
    
    # 1. Dataclasses and type hints
    user = User(id=1, name="John Doe", email="john@example.com")
    print(f"User: {user.to_dict()}")
    
    # 2. Decorators
    result = slow_operation()
    print(f"Operation result: {result}")
    
    # 3. Context managers
    new_user = create_user_with_transaction({"name": "Jane", "email": "jane@example.com"})
    print(f"Created user: {new_user.name}")
    
    # 4. Async operations
    async_user = await fetch_user_async(2)
    print(f"Async user: {async_user.name}")
    
    # 5. Generators
    fib_numbers = list(fibonacci_generator(10))
    print(f"Fibonacci: {fib_numbers}")
    
    # 6. Comprehensions
    print(f"Even squares: {even_squares}")
    print(f"User emails: {user_dict}")
    
    # 7. Closures
    double = create_multiplier(2)
    print(f"Double of 5: {double(5)}")
    
    # 8. Caching
    print(f"Fibonacci 10: {fibonacci(10)}")
    print(f"Fibonacci 10 (cached): {fibonacci(10)}")
    
    # 9. Enums
    print(f"User role: {UserRole.ADMIN.value}")
    
    # 10. Services (for FastAPI)
    db_service = DatabaseService("postgresql://localhost/mydb")
    user_service = UserService(db_service)
    
    user_request = UserCreateRequest(name="Alice", email="alice@example.com")
    response = await user_service.create_user(user_request)
    print(f"Created user response: {response.user.name}, Token: {response.token}")

    # Add context managers demo
    demonstrate_advanced_context_managers()

if __name__ == "__main__":
    asyncio.run(main())