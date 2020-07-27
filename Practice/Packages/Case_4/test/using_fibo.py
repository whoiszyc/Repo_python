import sys
# Here we would like to call a module from another directories
# Case 1 will not work since system only know the path til Python_Study_Practice

# Case 1
# try:
#     from Case_4.fibo.fibo1 import *
#     from Case_4.fibo.fibo2 import *
#     print(func_fibo1(10))
#     print(func_fibo2(10))
# except ModuleNotFoundError:
#     print("Oops!", sys.exc_info()[0], "occured.")
#     print('Case 1 will not work since system only know the path til Python_Study_Practice')
#     print('Proceed to Case 2')
#     print()
#     print()


# # Case 2
# # Thus, we need to add the direction to the system path
sys.path.append('C:\ZYC_Cloud\GitHub\Repo_python\Practice\Packages')
from Case_4.fibo.fibo1 import *
from Case_4.fibo.fibo2 import *
print(func_fibo1(10))
print(func_fibo2(10))
print()
print()


# Case 3
# Add path to "Case_4" will not work if Case_4.fibo.fibo1 is used.
# Always the parent directory will need adding into the system path.
# sys.path.append('C:\ZYC_Cloud\GitHub\Repo_python\Practice\Packages\Case_4')
# from fibo.fibo1 import *
# from fibo.fibo2 import *
# print(func_fibo1(10))
# print(func_fibo2(10))
# print()
# print()

