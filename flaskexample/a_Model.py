import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



def ModelIt(fromUser  = 'Default', births = []):
  in_month = len(births)
  print 'The number born is %i' % in_month

  mu, sigma = 100, 15
  x = mu + sigma*np.random.randn(10000)

  # the histogram of the data
  n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

  # add a 'best fit' line
  y = mlab.normpdf( bins, mu, sigma)
  l = plt.plot(bins, y, 'r--', linewidth=1)

  result = in_month
  if fromUser != 'Default':
    return result
    #return plt.show()
  else:
    return 'check your input'
