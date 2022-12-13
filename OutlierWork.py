# %%
ext_values = data[data.Total_Loss < data.Total_Loss.quantile(.005)]
# %%
X = ext_values.drop(columns=['Total_Loss', 'Trial', 'Timestep'])
#X = X.head(10000) # If wanting to reduce sample size

y = ext_values['Total_Loss'].values

# %%
X = pipeline.transform(X)
# %%
test_pred = lin_reg.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)

# %%
test_pred = model.predict(X)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y, test_pred)