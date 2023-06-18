import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

archive = pd.read_csv('data/archive_map.csv', header=None)

archive_np = archive.values
archive_np = np.delete(archive_np, 0, axis=1)
archive_np = np.delete(archive_np, 0, axis=0)

archive_np_flipud = np.flipud(archive_np)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(archive_np_flipud, origin='lower', extent=(0, 300000, 0, 150000), aspect='auto', cmap='plasma')
im.autoscale()

plt.xlabel("mean(congestion window size)")
plt.ylabel("std(conggestion window size)")
plt.colorbar(im)
plt.savefig("result_graph/archivemap.png")