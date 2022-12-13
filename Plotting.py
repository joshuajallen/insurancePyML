# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for i in data.columns:
    plt.figure()
    plt.hist(data[i], bins = 100)
# %%
# 78 and # 79 for paper

residuals_ln = (abs(y_test) - abs(np.squeeze(test_pred)))
residuals_nn = (abs(y_test) - abs(np.squeeze(test_pred_nn)))

# %%
plt.scatter(abs(residuals_ln), abs(residuals_nn), alpha = 0.2,c = 'grey')
plt.xlim(-0.1e9, 1.2e9)
plt.ylim(-0.1e9, 1e9)
plt.axline([0, 0], [1, 1], c = 'black', lw = 1)
plt.title("Comparison of residuals between models", y=1.08, fontsize=16)
plt.xlabel("Linear Regression Residuals",  fontsize=14)
plt.ylabel("ANN Residuals",  fontsize=14)
plt.figsize=(16, 12)
plt.savefig('Plots\\ResidualComparison.png', transparent=True, bbox_inches='tight', dpi = 300)
# %%
def qqplot(title, model, trainy, pred):
    residuals = (trainy - np.squeeze(pred))
    fig = sm.qqplot(residuals, alpha=0.01, markerfacecolor='k', markeredgecolor='k', line = 's') # “s” - standardized line, the expected 
    # order statistics are scaled by the standard deviation of the given sample and have the mean added to them
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title('QQ Plot for ' + title, fontsize=16)
    ax.xaxis.get_label().set_fontsize(14)
    ax.yaxis.get_label().set_fontsize(14)
    ax.get_lines()[1].set_color("grey")
    ax.get_lines()[1].set_lw(3)
    ax.get_lines()[1].set_alpha(0.7)
    ax.get_lines()[1].set_linestyle('--')
    ax.grid()
    fig.set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig('Plots\\QQ_Plot_' + title + '.png', transparent=True)
    plt.show()
    

# %% Plot QQ plot for Lin Reg
qqplot("Linear Regression", lin_reg, y_train, train_pred)
# %%
def plot_residuals(title, trainy, pred_train, testy, pred_test):
    residuals_train = (trainy - np.squeeze(pred_train))
    residuals_test = (testy - np.squeeze(pred_test))
    fig = plt.scatter(pred_train, residuals_train, c = 'black', label = 'Train')
    fig2 = plt.scatter(pred_test, residuals_test, c = 'grey', alpha = 0.7, label = 'Test')
    #plt.tight_layout()
    plt.title("Residuals for " + title + " Model", y=1.08, fontsize=16)
    plt.xlabel("Predicted Values",  fontsize=14)
    plt.ylabel("Residuals",  fontsize=14)
    plt.figsize=(8, 6)
    leg = plt.legend(loc="upper left")
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    #plt.savefig('Plots\\ResidualPlots' + title + '.png', transparent=True, bbox_inches='tight')
    plt.show()
    

# %% Plot QQ plot for Lin Reg
plot_residuals("Linear Regression", y_train, train_pred, y_test, test_pred)
# %%
from scipy.stats import norm
sns.distplot(data['Total_Loss'], color = 'black',  fit = norm, 
kde=False, fit_kws={"color": "k", "lw":1.5})
plt.title('Distribution of Capital across all scenarios', y=1.08, fontsize=16)
# Get the fitted parameters used by sns
(mu, sigma) = norm.fit(data['Total_Loss'])
# Legend and labels 
plt.legend(["Normal Distribution"])
plt.xlabel("Capital Value",  fontsize=14)
plt.ylabel("Density",  fontsize=14)
plt.figsize=(8, 6)
plt.xlim(-0.4e10, 0.6e10)
plt.savefig('Plots\\DistributionCapital.png', transparent=True, bbox_inches='tight', dpi = 300)

# %%
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
headers = ['Sample Size', 'Linear Regression', 'Artificial Neural Network']
df = pd.read_csv('SampleSizes.csv',names=headers)
# plot
plt.plot(df['Sample Size'],df['Linear Regression'])
plt.plot(df['Artificial Neural Network'])

plt.show()
# %%
sns.distplot(data[''], color = 'black',  fit = norm, 
kde=False, fit_kws={"color": "k", "lw":1.5})
plt.title('Distribution of dependent variable X37', y=1.08, fontsize=16)
plt.xlabel("X37 Value",  fontsize=14)
plt.ylabel("Frequency",  fontsize=14)
plt.figsize=(8, 6)
#plt.xlim(-0.4e10, 0.6e10)
plt.savefig('Plots\\DistributionX37.png', transparent=True, bbox_inches='tight', dpi = 300)

#plt.xlim(0, 2e8)
# %%
