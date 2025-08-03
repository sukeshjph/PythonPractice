from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Dict, Any, List, str

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