import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "hello.txt")

print("Script directory:", script_dir)
print("File path:", file_path)

try:
    with open(file_path, "r") as file:  # Use file_path instead of hardcoded path
        content = file.read()
        print(content)
except FileNotFoundError:
    print('File not found')