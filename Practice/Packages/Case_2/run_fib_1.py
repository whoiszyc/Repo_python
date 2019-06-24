# Methods to use Module fibo from Package Fibonacci

# First, if __init__.py has been correctly setup, then the package name can be used
# In __init__.py, one should write:
#   from .fibo import fib
#   from .fibo import fib2

import Fibonacci  # In this case, if the "__init__.py" is empty, then the system dose not know the path inside Fibonacci

print(Fibonacci.fibo.fib(10))
