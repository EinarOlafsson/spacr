"""Write a declared ZYX OME-TIFF plane by plane, like a live acquisition."""
import sys, time, json
import numpy as np, tifffile

path, planes, size, rate = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
times = int(sys.argv[5]) if len(sys.argv) > 5 else 1
yy, xx = np.mgrid[0:size, 0:size]
base = ((yy // 64 + xx // 64) % 7 * 40 + 100).astype(np.uint16)

def frames():
    for z in range(planes):
        plane = base.copy()
        plane[size // 4:size // 2, size // 4:size // 2] += 600 + z
        time.sleep(1.0 / rate)
        yield plane

start = time.time()
with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
    if times > 1:
        writer.write(frames(), shape=(times, planes // times, size, size), dtype=np.uint16,
                     metadata={'axes': 'TZYX'}, photometric='minisblack')
    else:
        writer.write(frames(), shape=(planes, size, size), dtype=np.uint16,
                     metadata={'axes': 'ZYX'}, photometric='minisblack')
closed = time.time()
print(json.dumps({'writer_start': start, 'writer_closed': closed}), flush=True)
