from lot_scheduling import make_tile_step, make_tiles_step
from utils import make_breakpoints

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

qmin = -20
qmax = 20
bp = make_breakpoints(qmin, qmax, 4, 0.02)
i0 = bp.index(0)

xs = np.arange(qmin, qmax+1, 1)
tiles = make_tiles_step(bp, xs)

matplotlib.rcParams['xtick.labelsize'] = 6
matplotlib.rcParams['ytick.labelsize'] = 6
matplotlib.rcParams['axes.labelsize'] = 6
fig, ax = plt.subplots()
fig.set_size_inches(3,1.25)
plt.subplot(2,1,1, adjustable='box', aspect=5)
plt.plot(xs, tiles[i0-2,:], 'r--', xs, tiles[i0+2,:], 'b-')
plt.subplot(2,1,2, adjustable='box', aspect=5)
plt.plot(xs, tiles[i0-1,:], 'r--', xs, tiles[i0], 'k:', xs, tiles[i0+1,:], 'b-')
plt.xlabel('Inventory level (items)')

fig.savefig("tiles.pdf", bbox_inches="tight", pad_inches=0)

