## EDA followed by Feature Importance

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Useful settings
#%matplotlib inline
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

data = pd.read_parquet('C:\\Users\\328504\\Documents\\Masters\\Proxy_Modelling\\Data\\insurance_dataset.gzip')
# %%
data.dtypes
# %%
data.shape
# %%
duplicate_rows_df = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
# %%
na_summary = data.isnull().sum()
na_summary[na_summary == 0]
# %%
data.head()
# %%
data.info()
# %%
data.describe()
# %%
data['Timestep'].unique()
# %%
data = data.drop(columns=['Trial', 'Timestep'])
# %%
data.head()
# %%
import numpy as np
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# %%
sns.distplot(data['Total_Loss'])
plt.title('Distribution of "Total_Loss" column', fontsize=18)
# %%
