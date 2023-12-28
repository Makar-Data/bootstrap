# Bootstrap

Bootstrap demonstrations from personal projects.

Variables to keep in mind:
- Bootstrap confidence level (boot_conf)
- Target statistic (statistic)
- Number or resamples (n_resamples)
- Method of bootstrap() (method)
- Confidence interval calculation method. Here the Efron method is used (based on percentiles)

Simple bootstrap
```Python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.stats import bootstrap  

data = np.array(df[column])  
  
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

data_1 = np.array(df.loc[df[groups] == 1)][column])
data_2 = np.array(df.loc[df[groups] == 2)][column])

boot_conf = 0.95
bootstrap1 = bootstrap(data=(data_1, ), statistic=np.mean, confidence_level=boot_conf, random_state=42, n_resamples=1000, method='percentile')
bootstrap2 = bootstrap(data=(data_2, ), statistic=np.mean, confidence_level=boot_conf, random_state=42, n_resamples=1000, method='percentile')

# Stat difference and Efron's confidence interval
difference = bootstrap2.bootstrap_distribution - bootstrap1.bootstrap_distribution
perc_low = ((1 - boot_conf) / 2) * 100
perc_high = (boot_conf * 100) + perc_low
ci_low = np.percentile(difference, perc_low)
ci_high = np.percentile(difference, perc_high)

# If the distribution is to the right of the red line - bootstrap 2 has bigger values. To the left, the bootstrap2.
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
