"""Evict and verify pages for a closed DuckDB database on Linux."""

import ctypes
import mmap
import os


def cold_file(path):
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size
        if not size:
            return
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            vector = (ctypes.c_ubyte * pages)()
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            if libc.mincore(address, size, vector):
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident = sum(v & 1 for v in vector)
            if resident:
                raise RuntimeError(f"Cold-cache check failed: {resident}/{pages} pages resident")
