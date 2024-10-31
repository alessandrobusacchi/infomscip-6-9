import pandas
import numpy as np
import matplotlib.pyplot as plt

filename = "Exp1_Results.csv"
df = pandas.read_csv(filename)

print(list(df.itertuples(index=False, name=None)))

samples = list(df.itertuples(index=False, name=None))
n_values, avg_misclassifications, std_devs = zip(*samples)
plt.errorbar(n_values, avg_misclassifications, yerr=std_devs, fmt='-o', capsize=5)
plt.xlabel('Number of points in S (density)')
plt.ylabel('Average misclassifications')
plt.title('Misclassification dependency on density')
plt.grid()
plt.show()