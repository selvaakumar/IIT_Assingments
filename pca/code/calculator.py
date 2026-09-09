import argparse




def add(x, y):
    """Add two numbers."""
    return x + y

def subtract(x, y):
    """Subtract two numbers."""
    return x - y

def multiply(x, y):
    """Multiply two numbers."""
    return x * y

def divide(x, y):
    """Divide two numbers."""
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple Terminal Calculator")
    
    # Define what inputs the terminal expects
    parser.add_argument("operation", choices=["add", "sub", "mul", "div"], help="The math operation")
    parser.add_argument("x", type=float, help="First number")
    parser.add_argument("y", type=float, help="Second number")
    
    args = parser.parse_args()
    

    try:
        if args.operation == "add":
            print(add(args.x, args.y))
        elif args.operation == "sub":
            print(subtract(args.x, args.y))
        elif args.operation == "mul":
            print(multiply(args.x, args.y))
        elif args.operation == "div":
            print(divide(args.x, args.y))
    except ValueError as e:
        print(f"❌ Error: {e}")
