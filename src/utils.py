def is_number(value):
    return isinstance(value, (int, float))

if __name__ == "__main__":
    value = input("Enter the value you want to check: ")
    try:
        num = float(value)
        print("Is number:", is_number(num))
    except ValueError:
        print("Is number:", is_number(value))
