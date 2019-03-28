# Merges two subarrays of arr[].
# First subarray is arr[l..m]
# Second subarray is arr[m+1..r]
def merge(khar, l, m, r):
    n1 = m - l
    n2 = r - m + 1

    # create temp arrays
    L = [0] * n1
    R = [0] * n2

    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = khar[l + i]

    for j in range(0, n2):
        R[j] = khar[m + j]

        # Merge the temp arrays back into arr[l..r]
    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = l  # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            khar[k] = L[i]
            i += 1
        else:
            khar[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there
    # are any
    while i < n1:
        khar[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], if there
    # are any
    while j < n2:
        khar[k] = R[j]
        j += 1
        k += 1

    return khar


# l is for left index and r is right index of the
# sub-array of arr to be sorted
def mergeSort(arr, l, r):
    if l < r:
        # Same as (l+r)/2, but avoids overflow for
        # large l and h
        m = int((l + (r - 1)) / 2)

        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
        return merge(arr, l, m + 1, r)


if __name__ == "__main__":
    arr = [9, 4, 5, 6, 3, 2, 10, 8]
    print(mergeSort(arr, 0, len(arr) - 1))