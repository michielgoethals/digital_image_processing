#dia 3: square wave

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 2.0*np.pi*1.5, 1000)

y1 = 1.3*np.sin(x)
y2 = 1.3*np.sin(3.0*x)/3.0
y3 = 1.3*np.sin(5.0*x)/5.0
y4 = 1.3*np.sin(11.0*x)/11.0
y5 = 1.3*np.sin(31.0*x)/31.0

square = signal.square(x)

fig, axs = plt.subplots(2, 5, figsize=(16,6))

t = np.linspace(0.0, 1.5, 1000)

axs[0,0].plot(t,y1)
axs[0,0].set_title("y = sin(1x)/1")
axs[0,1].plot(t,y2)
axs[0,1].set_title("y = sin(3x)/3")
axs[0,2].plot(t,y3)
axs[0,2].set_title("y = sin(5x)/5")
axs[0,3].plot(t,y4)
axs[0,3].set_title("y = sin(11x)/11")
axs[0,4].plot(t,y5)
axs[0,4].set_title("y = sin(31x)/31")

axs[1,0].plot(t,y1)
axs[1,0].plot(t,square, color='r')
axs[1,0].set_title("n = 1")

axs[1,1].plot(t,y1+y2)
axs[1,1].plot(t,square, color='r')
axs[1,1].set_title("n = 3")

axs[1,2].plot(t,y1+y2+y3)
axs[1,2].plot(t,square, color='r')
axs[1,2].set_title("n = 5")

axs[1,3].plot(t,y1+y2+y3+y4)
axs[1,3].plot(t,square, color='r')
axs[1,3].set_title("n = 11")

axs[1,4].plot(t,y1+y2+y3+y4+y5)
axs[1,4].plot(t,square, color='r')
axs[1,4].set_title("n = 31")

for ax in axs.flatten():
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlim(-0.1, 1.6)

plt.show()