import numpy as np
import pandas as pd

mydict = [{'a': 10, 'b': 20, 'c': 30, 'd': 40},{'a': 100, 'b': 200, 'c': 300, 'd': 400},{'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df=pd.DataFrame(mydict)

df.iloc[:,1]

df.iloc[:,1:2]

df.size

df.shape