
#(1) Use module inside a package

##(1-1) If the package is at the root
For example: we would like to use `formulation_1.py` in `run_1.py`
<pre>
_Restoration
    formulation_1.py
    run_1.py
</pre>
Then, we directly use `import module_name`

##(1-2) If the package is not at the root
For example: we would like to use `formulation_1.py` in `run_1.py`,
but `_Restoration` is in another folder
<pre>
_Resilience
    ...
    _Restoration
        formulation_1.py
        run_1.py
</pre>
Then two methods can be used.
### Method 1
The following import will only work if the modules (~~.py) files have been loaded in the "__init__.py".
`import package_name`

### Method 2
`import package_name.module_name`



#(2) Use module in different package

##(2-1) The parent directory of these packages will need adding into the system path

##(2-2) The absolute path to the specific module inside the parent directory will be used for import.

For example:
<pre>
Case 4
    fibo
        fibo1.py
        fibo2.py
    test
        using_fibo.py
</pre>
Then, in the file "using_fibo.py", the parent directory will need adding into the system path.

###Method 1
<pre>
sys.path.append('C:\ZYC_Cloud\GitHub\Python_Study_Practice\Practice\Packages\Case_4')
from fibo.fibo1 import *
from fibo.fibo2 import *
</pre>

###Method 2
<pre>
sys.path.append('C:\ZYC_Cloud\GitHub\Python_Study_Practice\Practice\Packages')
from Case_4.fibo.fibo1 import *
from Case_4.fibo.fibo2 import *
<pre>