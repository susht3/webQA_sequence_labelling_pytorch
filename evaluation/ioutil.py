from gzip import GzipFile
import os

# for python 2.6
class GzipFileHack(GzipFile):
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.close()

def open_file(filename, *args1, **args2):
    if filename.endswith('.gz'):
        return GzipFileHack(filename, *args1, **args2)
    else:
        return open(filename, *args1, **args2)

def mkdirs(dirname, mode=0755):
    if os.path.exists(dirname): return
    os.makedirs(dirname, mode)

def is_not_empty(filename):
    if not os.path.exists(filename): return False
    return os.path.getsize(filename) != 0
