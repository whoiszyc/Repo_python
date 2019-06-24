
(1) Use module inside a package

The following import will only work if the modules (~~.py) files have been loaded in the "__init__.py".
import package_name

Otherwise, use:
import package_name.module_name



(2) Use module in different package

(2-1) The parent directory of these packages will need adding into the system path

(2-2) The absolute path to the specific module inside the parent directory will be used for import.

For example:
Case 4
    fibo
        fibo1.py
        fibo2.py
    test
        using_fibo.py
        
Then, in the file "using_fibo.py", the parent directory will need adding into the system path.

Alternative 1:
sys.path.append('C:\ZYC_Cloud\GitHub\Python_Study_Practice\Practice\Packages\Case_4')
from fibo.fibo1 import *
from fibo.fibo2 import *

Alternative 1:
sys.path.append('C:\ZYC_Cloud\GitHub\Python_Study_Practice\Practice\Packages')
from Case_4.fibo.fibo1 import *
from Case_4.fibo.fibo2 import *