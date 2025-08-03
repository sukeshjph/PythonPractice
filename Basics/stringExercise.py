# Write a Python program that:

# Asks the user to enter their full name (with spaces)

# Prints the name in uppercase

# Prints the initials (e.g., "S D" for "Sukesh Dash")

# Prints the number of characters (excluding spaces)

full_name = input('Please enter your name')

print(f"Uppercase: {full_name.upper()}")

# print(fnameLastname[0][0])
# print(fnameLastname[1][0])
parts = full_name.strip().split()
print(full_name.strip().split())
initials = " ".join([p[0].upper() for p in parts])
print(initials)


