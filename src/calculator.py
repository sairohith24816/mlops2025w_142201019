from utils import is_number

def add(a, b):
    if is_number(a) and is_number(b):
        return a + b
    raise ValueError("Invalid input: Both inputs must be numbers.")

if __name__ == "__main__":
    print("welcome to the calculator")
    a = input("Enter first number: ")
    b = input("Enter second number: ")
    try:
        a_val = float(a)
        b_val = float(b)
        result = add(a_val, b_val)
        print(f"Addition Result: {result}")
    except ValueError:
        print("Please enter numbers only.")