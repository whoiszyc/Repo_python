# Test mapping unpacking operator (*)
# **kwargs will be mapping types, that is, dictionary

def add_person_details(ssn, surname, **kwargs):
    print("SSN =", ssn)
    print(" surname =", surname)
    for key in sorted(kwargs):
        print(" {0} = {1}".format(key, kwargs[key]))


add_person_details(83272171, "Luther", forename="Lexis", age=47)

