# age = 18

# if age < 18:
#   print('You are minor')
# elif age == 18:
#   print('You are adult')
# else:
#   print('You are old')

# value = 3
# match value:
#   case 1:
#     print('One')
#   case 2:
#     print('Two')
#   case 3:
#     print('Three')
#   case _:
#     print('Something else')


# fruits = ['apple', 'banana', 'cherry']
# for fruit in fruits:
#   print(fruit)

# for i in range(5):
#   print(i)  

# count = 0

# while count < 5:
#   print(count);
#   count+=1 

# for i in range(10):
#   # if i == 5:
#   #   break
#   print(i)

# for i in range(1,11):
#   for j in range(1, 11):
#     product = i * j
    
#     if(product % 3 != 0):
#       print(f'{i} * {j} = {product}')

# ===== RANGE METHOD EXPLANATION =====
print("=== RANGE METHOD EXPLANATION ===")
print()

# 1. range(stop) - starts from 0, goes up to (but not including) stop
print("1. range(5) - starts from 0, goes up to 4:")
for i in range(5):
    print(f"  {i}", end=" ")
print("\n")

# 2. range(start, stop) - starts from 'start', goes up to (but not including) 'stop'
print("2. range(2, 7) - starts from 2, goes up to 6:")
for i in range(2, 7):
    print(f"  {i}", end=" ")
print("\n")

# 3. range(start, stop, step) - starts from 'start', goes up to 'stop', increments by 'step'
print("3. range(0, 10, 2) - starts from 0, goes up to 9, increments by 2:")
for i in range(0, 10, 2):
    print(f"  {i}", end=" ")
print("\n")

# 4. Negative step (counting backwards)
print("4. range(10, 0, -1) - counts backwards from 10 to 1:")
for i in range(10, 0, -1):
    print(f"  {i}", end=" ")
print("\n")

# 5. Converting range to list
print("5. Converting range to list:")
numbers = list(range(1, 6))
print(f"  list(range(1, 6)) = {numbers}")
print()

# 6. Range with different step values
print("6. Different step values:")
print("  range(0, 20, 3):", end=" ")
for i in range(0, 20, 3):
    print(f"{i}", end=" ")
print()

print("  range(10, -5, -2):", end=" ")
for i in range(10, -5, -2):
    print(f"{i}", end=" ")
print("\n")

# 7. Common use cases
print("7. Common use cases:")
print("  - Counting from 0 to n-1: range(n)")
print("  - Counting from 1 to n: range(1, n+1)")
print("  - Even numbers: range(0, n, 2)")
print("  - Odd numbers: range(1, n, 2)")
print("  - Reverse counting: range(n, 0, -1)")
print()

# 8. Range in your multiplication table context
print("8. How range works in your multiplication table:")
print("  range(1, 11) creates: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
print("  This gives you numbers 1 through 10 for both rows and columns")
print()

# 9. Memory efficiency
print("9. Memory efficiency:")
print("  range() is memory efficient - it doesn't create the full list in memory")
print("  It generates numbers on-demand as you iterate through them")
print("  This is why range(1000000) doesn't use much memory")


