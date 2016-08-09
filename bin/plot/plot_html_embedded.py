import numpy as np
import math

import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.serif": [],
    "pgf.texsystem": "pdflatex"
}
mpl.rcParams.update(pgf_with_rc_fonts)

import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 7000, 1000)

sigma = 1.7643
mu = 7.90272
html = stats.lognorm(s=sigma, scale=math.exp(mu))

sigma_embedded = 2.17454
mu_embedded = 7.51384
embedded = stats.lognorm(s=sigma_embedded, scale=math.exp(mu_embedded))

# td: label for red and black
plt.subplot(121)

plt.plot(x, html.pdf(x), 'k', label="HTML page sizes")
plt.plot(x, embedded.pdf(x), 'r', label="embedded object sizes")
plt.xlabel('size (Byte)')
plt.ylabel('probability')
plt.title('probability density of sizes')
plt.legend()

x2 = np.linspace(0, 40, 100)

kappa=0.141385
theta=40.3257
num_embedded = stats.gamma(kappa, scale=theta)

plt.subplot(122)
plt.plot(x2, num_embedded.pdf(x2))
plt.xlabel('number of embedded elements')
plt.show()
