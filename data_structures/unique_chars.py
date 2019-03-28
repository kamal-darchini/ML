import time

def unique_chars(text: str) -> bool:
    now = time.time()
    d = {}
    for chr in text:
        d.update({chr: None})

    if len(d) == len(text):
        print("It took ", time.time() - now, " seconds to run this code.")
        return True
    else:
        print("It took ", time.time() - now, " seconds to run this code.")
        return False


if __name__ == "__main__":
    print(unique_chars("Helo!Helo!Helo!Helo!Helo!"))
