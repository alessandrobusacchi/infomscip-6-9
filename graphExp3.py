import pandas
import numpy as np
import matplotlib.pyplot as plt

filename = "Exp3_Results.csv"
df = pandas.read_csv(filename)

print(list(df.itertuples(index=False, name=None)))

samples = list(df.itertuples(index=False, name=None))
ratio, perimeter, avg_misclassifications, std_devs = zip(*samples)

plt.errorbar(ratio, avg_misclassifications, yerr=std_devs, fmt='-o', capsize=5)
plt.xlabel('Triangle ratio')
plt.ylabel('Average misclassifications')
plt.title('Misclassification dependency on triangle ratio')
plt.grid()

idx = np.where(np.array(ratio) == 1)[0][0]
highlighted_ratio = ratio[idx]
highlighted_value = avg_misclassifications[idx]
plt.scatter(highlighted_ratio, highlighted_value, color='red', zorder=5, label=f'Original triangle: ratio={highlighted_ratio}, value={highlighted_value:.2f}')
plt.legend()
plt.show()