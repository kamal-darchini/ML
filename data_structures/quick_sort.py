import random
import time


def partition(arr, l, h):
    pivot_value = arr[l]
    i = l + 1
    j = h

    done = False
    while not done:

        while i <= j and arr[i] <= pivot_value:
            i = i + 1

        while arr[j] >= pivot_value and j >= i:
            j = j - 1

        if j < i:
            done = True
        else:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp

    temp = arr[l]
    arr[l] = arr[j]
    arr[j] = temp

    return j


def _quick_sort(arr, l, h):
    if l < h:
        pivot_loc = partition(arr, l, h)
        _quick_sort(arr, l, pivot_loc - 1)
        _quick_sort(arr, pivot_loc + 1, h)

    return arr


def quick_sort(arr, l, h):
    random.shuffle(arr)
    return _quick_sort(arr, l, h)


if __name__ == "__main__":
    now = time.time()
    a = [1, 2, 0, 0, 0, 0, 7, 10]
    print(quick_sort(a * 100, 0, 8 * 100 - 1))
    print("It took ", time.time() - now, "seconds.")
