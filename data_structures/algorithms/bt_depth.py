from data_structures.bst import bst


def max_depth(tree):

    if None == tree:
        return 0
    else:
        return 1 + max(max_depth(tree.left_child), max_depth(tree.right_child))


if __name__ == "__main__":
    BST = bst()
    data =[1, 3, 5, 2, 10, 6, 7, 8, 8]
    for d in data:
        BST.insert(d)

    print(max_depth(BST.root))