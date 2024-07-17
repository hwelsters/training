import time

def log(*argv):
    localtime = time.localtime()
    print(f"[{localtime.tm_hour:02}:{localtime.tm_min:02}:{localtime.tm_sec:02}]", end=" ")
    for arg in argv:
        print(arg, end=" ")
    print("")