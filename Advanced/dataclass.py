from dataclasses import dataclass, field
from typing import Optional, List, datetime

@dataclass
class Person():
    name: str
    age: int
    height: float
    email: Optional[str] = None

print(Person("Sukesh", 45, 5.11))

@dataclass
class Product:
    name: str                                       # Required
    price: float                                    # Required
    description: Optional[str] = None               # Optional
    category: Optional[str] = None                  # Optional
    tags: List[str] = field(default_factory=list)  # Optional list (empty by default)
    metadata: dict = field(default_factory=dict)   # Optional dict (empty by default)


print(Product("Mark", 123456))

# Dataclass with field() for complex defaults
@dataclass
class User:
    username: str
    email: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True