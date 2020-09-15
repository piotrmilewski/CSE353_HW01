import numpy as np
import math
from hw1 import question2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm


# basically the same function except it gives us predict function capability
def question2bHelper(x, y):
    init1 = np.flipud(x[0: 7])
    init2 = np.flipud(y[0: 7])
    init = np.concatenate((init1, init2))
    linRegX = np.array([init])
    index = 8
    while index < len(x):
        data1 = np.flipud(x[index - 7: index])
        data2 = np.flipud(y[index - 7: index])
        data = np.concatenate((data1, data2))
        linRegX = np.vstack([linRegX, data])
        index = index + 1
    linRegY = y[7:len(y)]
    reg = LinearRegression().fit(linRegX, linRegY)

    init1 = np.flipud(x[0: 7])
    init2 = np.flipud(y[0: 7])
    init = np.concatenate((init1, init2))
    retNewY = reg.predict(np.array([init]))
    index = 8
    while index < len(x):
        data1 = np.flipud(x[index - 7: index])
        data2 = np.flipud(y[index - 7: index])
        data = np.concatenate((data1, data2))
        tempNewY = reg.predict(np.array([data]))
        retNewY = np.concatenate((retNewY, tempNewY))
        index = index + 1
    return retNewY


inputFromCSV = np.genfromtxt('question2_time_series_covid19.csv', delimiter=',',
                             skip_header=1, dtype=None, encoding=None,
                             converters={0: lambda x: -1})  # set first column to -1

inputFromCSV = np.delete(inputFromCSV, 0, 1)  # remove first column

xInput = inputFromCSV[0]
yInput = inputFromCSV[1]

retVal = question2(xInput, yInput)

# a ----------------------------------------------------------------------------

print(retVal.get('w'))
print(retVal.get('b'))

# b ----------------------------------------------------------------------------

newY = question2bHelper(xInput, yInput)
oldY = yInput[7:]

xAxis = np.arange(1, 214)

plt.scatter(xAxis, oldY, c="black", s=10)

plt.scatter(xAxis, newY, c="red", s=10)

plt.xlabel("Days since start of outbreak")
plt.ylabel("Death Count")

plt.savefig('scatter_curve.png')
plt.show()
