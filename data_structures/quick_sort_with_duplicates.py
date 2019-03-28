import random
import time


def partition(arr, l, h):
    pivot_value = arr[l]
    i = l + 1
    j = h
    min_pivot = l

    while i <= j:

        if arr[i] < pivot_value:
            tmp = arr[i]
            arr[i] = arr[min_pivot]
            arr[min_pivot] = tmp
            min_pivot += 1
            i += 1

        elif arr[i] > pivot_value:
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
            j -= 1

        else:
            i += 1

    return min_pivot, i - 1


def _quick_sort(arr, l, h):
    if l < h:
        min_pivot, max_pivot = partition(arr, l, h)
        _quick_sort(arr, l, min_pivot - 1)
        _quick_sort(arr, max_pivot + 1, h)

    return arr


def quick_sort(arr, l, h):
    random.shuffle(arr)
    return _quick_sort(arr, l, h)


if __name__ == "__main__":
    now = time.time()
    a = [1, 2, 0, 0, 0, 0, 7, 10]
    print(quick_sort(a * 100, 0, 8 * 100 - 1))
    print("It took ", time.time() - now, "seconds.")
