import os
import random
import time
from multiprocessing import Pool


def error_handler(e):
    print('error')
    print(dir(e), "\n")
    print("-->{}<--".format(e.__cause__))

def long_time_task(name, s):
    print('Run task %s (%s)...' % (name, os.getpid()), s)
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,i), error_callback=error_handler)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')