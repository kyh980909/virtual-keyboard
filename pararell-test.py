import cv2
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
import multiprocessing as mp

def work(id, start, end, result):
    data = pd.DataFrame(columns=['x','y'])
    for x in range(start, end):
        for y in range(200):
            temp = pd.DataFrame({'x':[x], 'y':[y]})
            data = data.append(temp, ignore_index=True)

    result.put(data)
    
    return

if __name__ == '__main__':
    result = Queue()
    procs = []
    
    for i, x in enumerate(range(5)):
        proc = Process(target=work, args=(i, int((200/10*x)), int((200/10*(x+1))), result))
        procs.append(proc)
        proc.start()
        print('proc:',proc.ident)

    for proc in procs:
        proc.join()
        print(proc.is_alive())
    
    print('end join')
 
    test_data = pd.DataFrame(columns=['x','y'])
    while True:
        if result.empty():
            print('empty')
            break
        tmp = result.get()
        print(tmp)
        test_data.append(tmp)

    print(test_data)

    result.close()
    result.join_thread()
