import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.linspace(0, 7000, 1000)

sigma = 1.7643
mu = 7.90272
html = stats.lognorm(s=sigma, scale=math.exp(mu))

sigma_embedded = 2.17454
mu_embedded = 7.51384
embedded = stats.lognorm(s=sigma_embedded, scale=math.exp(mu_embedded))

# td: label for red and black
plt.subplot(211)

plt.plot(x, html.pdf(x), 'k', label="HTML pages")
plt.plot(x, embedded.pdf(x), 'r', label="embedded objects")
plt.title('Probability Density of Object Sizes')
plt.xlabel('object size [Byte]')
plt.ylabel('probability')
plt.legend()

x2 = np.linspace(0, 40, 1000)

kappa=0.141385
theta=40.3257
num_embedded = stats.gamma(kappa, scale=theta)

plt.subplot(212)
plt.plot(x2, num_embedded.pdf(x2))
plt.title('Probability Density of Number of Embedded Elements')
plt.xlabel('number of elements')
plt.ylabel('probability')
plt.ylim(0, 1)
plt.tight_layout()

## code to use pgf backend (\input(figname.fig) in latex, see in figname.fig)
#import matplotlib as mpl
#mpl.use("pgf")
# pgf_with_rc_fonts = {
#     "font.serif": [],
#     "pgf.texsystem": "pdflatex"
# }
# mpl.rcParams.update(pgf_with_rc_fonts)
# plt.savefig('/tmp/fig_html_embedded.pgf')
plt.savefig('/tmp/fig_html_embedded.pdf')

plt.show()
