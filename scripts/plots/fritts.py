
import matplotlib
import matplotlib.pylab as plt
import numpy as np

from numpy import pi, sqrt, exp, log

pi = np.pi


# make a box

x = np.linspace(0.0, 3.0, 101)
y = 1.0/(x*sqrt(2*pi)) * exp(-log(x)**2/2.0)

plt.plot(3.0-x, y)


plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

ax = plt.gca()
#ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 2.5]))
#ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(['forest interior', 'semi-arid\nforest border']))



ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

ax.annotate('forest interior', xy=(0.5,0.1), ha='center', va='center')
ax.annotate('semi-arid\nforest border', xy=(2.5,0.1),  ha='center', va='center')
ax.annotate('complacent', xy=(0.2, 0.7), xytext=(0.6, 0.7), va='center',
            arrowprops=dict(arrowstyle="->"))

plt.xlim([0.0, 2.925])
plt.ylim([0.0, 0.8])

plt.show()



