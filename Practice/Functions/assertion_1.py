# Assertion test
# syntax: assert boolean_expression, optional_expression

# If the boolean_expression evaluates to False an AssertionError exception is raised.
# If the optional optional_expression is given, it is used as the argument
# to the AssertionError exception—this is useful for providing error messages.

def product(*args): # optimistic
    result = 1
    for arg in args:
        result *= arg
    assert result, "0 argument"
    return result

print(product(1,0,5))



