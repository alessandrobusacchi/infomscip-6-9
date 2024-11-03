import pandas
import numpy as np
import matplotlib.pyplot as plt

filename = "Exp2_Results.csv"
df = pandas.read_csv(filename)

print(list(df.itertuples(index=False, name=None)))

samples = list(df.itertuples(index=False, name=None))
f_values, avg_misclassifications, std_devs = zip(*samples)
plt.errorbar(f_values, avg_misclassifications, yerr=std_devs, fmt='-o', capsize=5)
plt.xlabel('Outlier fraction (f)')
plt.ylabel('Average misclassifications')
plt.title('Misclassification dependency on outlier fraction')
plt.grid()
plt.show()