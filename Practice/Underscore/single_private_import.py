from single_private import *

Prefix = Prefix()
print(Prefix.public)
print(Prefix._private)

# for function defined in class, it seems no difference
Prefix.public_api()
Prefix._private_api()

# for function defined in modules, it has a difference
public_api()
_private_api


