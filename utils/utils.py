import os

__all__ = ['vstack', 'folder']

def vstack(arr_2):
    output = []
    for i in arr_2:
        output.extend(i)
    return output

def folder(dir):
    for i in dir:
        if not os.path.isdir(i):
            os.makedirs(i)