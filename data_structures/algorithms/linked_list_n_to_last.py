from data_structures.linked_list import LinkedLIst


def n_to_last(n, linked_l):
    if linked_l.count() < n:
        return None

    q = linked_l.peek()
    p = linked_l.peek()
    c = 0
    while p != None:
        p = p.next
        c += 1
        if c > n and p!= None:
            q = q.next
    return q.val



if __name__ == "__main__":
    ll = LinkedLIst()
    for i in range(10):
        ll.add_last(i)
    print(ll.traverse())

    print(n_to_last(3, ll))