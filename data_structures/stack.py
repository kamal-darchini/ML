class Stack:

    def __init__(self):
        self.s = []
        self.n = 0

    def push(self, val):
        self.s.append(val)
        self.n += 1

    def pop(self):
        if self.s != []:
            self.n -= 1
        return self.s.pop()


if __name__ == "__main__":
    s = Stack()
    for i in range(10):
        s.push(i)

    print(s.n)
    s.pop()
    print(s.n)
    print(s.s)

