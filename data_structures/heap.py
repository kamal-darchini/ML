import heapq


class MinHeap:

    def __init__(self):
        self.h = []

    def push(self, item):
        heapq.heappush(self.h, item)

    def pop(self):
        return heapq.heappop(self.h)

    def peek(self):
        return self.h[0]

    def __getitem__(self, item):
        return self.h[item]

    def __len__(self):
        return len(self.h)


class MaxHeap(MinHeap):

    def push(self, item):
        heapq.heappush(self.h, Comparator(item))

    def __getitem__(self, item):
        return self.h[item].value


class Comparator:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __gt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value >= other.value

    def __ge__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        return self.value == other.value


if __name__ == "__main__":
    import time
    now = time.time()
    a = [1, 2, 0, 0, 0, 0, 7, 10, 1]
    min_h = MinHeap()
    for item in a:
        min_h.push(item)
    print(min_h.h)

    max_h = MaxHeap()
    for item in a:
        max_h.push(item)
    print([x.value for x in max_h.h])

    while True:
        try:
            print(max_h.pop().value)
        except IndexError:
            break


