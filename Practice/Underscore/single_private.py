class Prefix:
    def __init__(self):
        self.public = 10
        self._private = 12

    def public_api(self):
        print("public api")

    def _private_api(self):
        print("private api")


def public_api():
    print("public api")

def _private_api():
    print("private api")

