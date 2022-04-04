# us-adult-census-income-classification
the data in the "input" folder.

the models are in the "src" folder.

the models are saved in the "models" folder.

the 'notebooks" is used for data exploration and visualisation.

to create kfolds using a stratified classifer, "cd" into the src folder & run:
```bash
python create_folds.py
```
to run the logistic regression model, "cd" into the src folder & run:
```bash
python ohe_logres.py
```
to run the xgb boost model, "cd" into the src folder & run:
```bash
python lbl_xgb.py
```
