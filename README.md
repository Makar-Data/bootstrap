# Bootstrap

Bootstrap demonstrations based on eCommerce behavior dataset: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Simple bootstrap
```Python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import bootstrap  
  
df = pd.read_pickle('sales_data.pkl')  
data = np.array(df['price'])  
  
boot_conf = 0.95  
bootstrap_ci = bootstrap(data=(data, ), statistic=np.mean, confidence_level=boot_conf, random_state=42, n_resamples=1000, method='percentile')  
ci = bootstrap_ci.confidence_interval  

fig, ax = plt.subplots()  
ax.hist(bootstrap_ci.bootstrap_distribution, bins=25)  
ax.axvline(ci[0], color='red', label='Lower: {}'.format(round(ci[0], 2)))  
ax.axvline(ci[1], color='red', label='Upper: {}'.format(round(ci[1], 2)))  
  
ax.set_title('Bootstrap Distribution with {}% CI'.format(round(boot_conf*100)))  
ax.set_xlabel('statistic')  
ax.set_ylabel('frequency')  
  
plt.legend(shadow=True)  
plt.tight_layout()  
plt.show()
```
![image](https://github.com/Makar-Data/bootstrap/assets/152608115/c97a10f6-b24c-412d-8af1-81d575ed07f8)


Bootstrap hypothesis testing
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from sklearn import datasets

# df = pd.read_pickle('sales_data.pkl')
# data = np.array(df['price'])

iris_data = datasets.load_iris()
mydata = pd.DataFrame(data = iris_data.data, columns = iris_data.feature_names)
mydata['target'] = iris_data.target
dataportion = mydata[['sepal length (cm)', 'target']]

sepal_versi = dataportion.loc[dataportion['target'] == 1]['sepal length (cm)']
sepal_virgi = dataportion.loc[dataportion['target'] == 2]['sepal length (cm)']

boot_conf = 0.95
bootstrap1 = bootstrap(data=(sepal_virgi, ), statistic=np.mean, confidence_level=boot_conf, random_state=42, n_resamples=1000, method='percentile')
bootstrap2 = bootstrap(data=(sepal_versi, ), statistic=np.mean, confidence_level=boot_conf, random_state=42, n_resamples=1000, method='percentile')

# Разница и доверительный интервал Эфрона
difference = bootstrap2.bootstrap_distribution - bootstrap1.bootstrap_distribution
perc_low = ((1 - boot_conf) / 2) * 100
perc_high = (boot_conf * 100) + perc_low
ci_low = np.percentile(difference, perc_low)
ci_high = np.percentile(difference, perc_high)

print(bootstrap1.confidence_interval)
print(bootstrap2.confidence_interval)

# Если правее - то bootstrap2 больше. Если левее, то bootstrap1
fig, ax = plt.subplots()
ax.hist(difference, bins=25)
ax.axvline(x=0, color='red', label='Zero: 0.00')
ax.axvline(ci_low, color='blue', label='Lower: {}'.format(round(ci_low, 2)))
ax.axvline(ci_high, color='blue', label='Upper: {}'.format(round(ci_high, 2)))

ax.set_title('Bootstrap Statistic Difference with {}% CI'.format(round(boot_conf*100)))
ax.set_xlabel('statistic difference')
ax.set_ylabel('frequency')

plt.legend(shadow=True)
plt.tight_layout()
plt.show()
```
![image](https://github.com/Makar-Data/bootstrap/assets/152608115/a3e4cee9-7404-42e8-97bb-599b0f457863)
