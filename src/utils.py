class Pipe:
    def __init__(self, value):
        self.value = value

    def __or__(self, f):
        return Pipe(f(self.value))
    
    @classmethod
    def get_value(cls, pipe):
        return pipe.value