def is_number(value):
    """Check if the given value is a number (int or float)."""
    return isinstance(value, (int, float))

if __name__ == "__main__":
    print("Welcome to the Utils module")
    print("Is 5 a number?", is_number(5))
    print("Is 5.0 a number?", is_number(5.0))
    print("Is 'hello' a number?", is_number("hello"))
