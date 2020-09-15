import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from hw1 import question1

inputFromCSV = np.genfromtxt('question1_covid_metadata.csv', delimiter=',',
                             skip_header=1, dtype=None, encoding=None,
                             converters={1: lambda x: 1 if x == 'F' else 0, 2: lambda x: 1 if x == 'Y' else 0})

featureMatrix = inputFromCSV[:, [0, 1]]
labelsVector = inputFromCSV[:, 2]

result = question1(featureMatrix, labelsVector)

print(result)

x1 = np.arange(0, 120, 0.01)
y1 = norm.pdf(x1, result.get('mu0')[0], math.sqrt(result.get('var0')[0]))

plt.plot(x1, y1, label="line 1", color="black")

x2 = np.arange(0, 120, 0.01)
y2 = norm.pdf(x2, result.get('mu1')[0], math.sqrt(result.get('var1')[0]))

plt.plot(x2, y2, label="line 1", color="blue")

plt.xlabel("Age")
plt.ylabel("Density")

plt.savefig('age_curve.png')
plt.show()
