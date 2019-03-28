from data_structures.stack import Stack


class SetofStacks:
    def __init__(self, threshold):
        self.threshold = threshold
        self.set_of_stacks = []

    def push(self, item):
        if self.set_of_stacks == []:
            self.set_of_stacks.append(Stack())
        if self.set_of_stacks[-1].n > self.threshold:
            self.set_of_stacks.append(Stack())
        self.set_of_stacks[-1].push(item)

    def pop(self):
        if self.set_of_stacks == []:
            return None
        self.set_of_stacks[-1].pop()
        if self.set_of_stacks[-1].n == 0:
            self.set_of_stacks.pop()


if __name__ == "__main__":
    ss = SetofStacks(2)

    for i in range(10):
        ss.push(i)

    print(len(ss.set_of_stacks))
    ss.pop()
    ss.pop()
    ss.pop()
    ss.pop()
    print(len(ss.set_of_stacks))

    for i in range(10):
        ss.push(i)
    print(len(ss.set_of_stacks))