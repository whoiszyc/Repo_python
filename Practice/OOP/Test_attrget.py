class Foo: pass
f = Foo()
f.bar = Foo()
f.bar.baz = Foo()
f.bar.baz.quux = "Found me!"

import operator
print(operator.attrgetter("bar.baz.quux")(f))