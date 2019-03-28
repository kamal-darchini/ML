class Node:

    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None


class bst:

    def __init__(self):
        self.root = None

    def insert(self, value):
        if None == self.root:
            self.root = Node(value)
        else:
            self._insert_node(self.root, value)

    def _insert_node(self, current_node, value):
        if None == current_node:
            current_node = Node(value)
            return
        if value <= current_node.value:
            if None == current_node.left_child:
                current_node.left_child = Node(value)
            else:
                self._insert_node(current_node.left_child, value)
        else:
            if None == current_node.right_child:
                current_node.right_child = Node(value)
            else:
                self._insert_node(current_node.right_child, value)

    def find(self, value):
        return self._find_node(self.root, value)

    def _find_node(self, current_node, value):
        if None == current_node:
            return False
        if current_node.value == value:
            return True
        if value < current_node.value:
            return self._find_node(current_node.left_child, value)
        else:
            return self._find_node(current_node.right_child, value)


if __name__ == "__main__":
    BST = bst()
    data =[1, 3, 5, 2, 10, 6, 7, 8, 8]
    for d in data:
        BST.insert(d)

    print(BST.find(8))
    print(BST.find(12))
    print(BST.find(2))
    print(BST.find(80))

