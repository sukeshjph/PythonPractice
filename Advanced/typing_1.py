from typing import List, Dict , Union, Any , Optional, TypeVar , Generic

# Type Hints

def get_user_name(user_Id: int) -> str:
    return f"The user name is {user_Id}"

def process_users(users:List[Dict[str, Any]]) -> Optional[List[str]]:
    if not users:
        return None
    return [user.get("name", "") for user in users]


def handle_id(user_id: Union[int, str]) -> str:
    return str(user_id)


T = TypeVar('T')

class ApiResponse(Generic[T]):
    def __init__(self, data: T, status: int = 20):
        self.data = data
        self.status = status


print(get_user_name(10))

print(handle_id(34))

users_data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]






