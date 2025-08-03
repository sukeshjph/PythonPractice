import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class User:
    id: int
    username: str
    email: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

async def fetch_user_async(user_id: int) -> User:
    # Simulate async database call
    await asyncio.sleep(0.1)
    return User(id=user_id, username=f"User {user_id}", email=f"user{user_id}@example.com")

# Async with multiple operations
# async def get_user_with_posts(user_id: int) -> Dict[str, Any]:
#     # Parallel execution (like Promise.all)
#     user_task = fetch_user_async(user_id)
#     posts_task = fetch_user_posts_async(user_id)
    
#     user, posts = await asyncio.gather(user_task, posts_task)
    
#     return {
#         "user": user.to_dict(),
#         "posts": posts
#     }

# async def fetch_user_posts_async(user_id: int) -> List[Dict[str, Any]]:
    await asyncio.sleep(0.1)
    return [
        {"id": 1, "title": "Post 1", "user_id": user_id},
        {"id": 2, "title": "Post 2", "user_id": user_id}
    ]


async def main():
    user = await fetch_user_async(user_id=1234)
    print(user)
    
    # Or call multiple users concurrently
    # users = await asyncio.gather(
    #     fetch_user_async(1),
    #     fetch_user_async(2),
    #     fetch_user_async(3)
    # )
    
# Run the main function
asyncio.run(main())