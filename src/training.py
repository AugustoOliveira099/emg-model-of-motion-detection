from RandomForestModelTrainer import RandomForestModelTrainer

rf_model = RandomForestModelTrainer()
df = rf_model.load_features_to_dataframe()
X_train, X_val, y_train, y_val = rf_model.split_data(df, test_size=0.2)
X_val, X_test, y_val, y_test = rf_model.split_data(X_val.join(y_val), test_size=0.5)
rf_model.train(X_train, y_train)
print(f"accuracy on the validation set: {rf_model.evaluate(X_val, y_val)}")
print(f"accuracy on the test set: {rf_model.evaluate(X_test, y_test)}")
rf_model.save_model()
