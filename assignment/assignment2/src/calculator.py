from utils import is_number

def add(a, b):
    """Add two numbers."""
    if is_number(a) and is_number(b):
        return a + b
    raise ValueError("Invalid input: Both arguments must be numbers.")

if __name__ == "__main__":
    print("welcome to the calculator")
    print("Addition Example: 5 + 7 =", add(5, 7))