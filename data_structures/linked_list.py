class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedLIst:

    def __init__(self):
        self.ll = []

    def add_last(self, item):
        node = Node(item)
        if self.ll != []:
            self.ll[-1].next = node
        self.ll.append(node)

    def remove(self, index):
        self.ll[index - 1].next = self.ll[index + 1]
        self.ll.pop(index)

    def count(self):
        return len(self.ll)

    def traverse(self):
        rslt = []
        for item in self.ll:
            rslt.append(item.val)
        return rslt

    def peek(self):
        return self.ll[0]


if __name__ == "__main__":
    ll = LinkedLIst()
    for i in range(10):
        ll.add_last(i)
    ll.add_last(20)
    ll.add_last(0)

    print(ll.count())

    print(ll.traverse())
