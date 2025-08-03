def decorator(func):
    def wrapper(*args, **kwargs):
        print('The function is wrapped')
        return func(*args, **kwargs)
    return wrapper 


@decorator
def add(a,b):
   return a+b

print(add(5,6))