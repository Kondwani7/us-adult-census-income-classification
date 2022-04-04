import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/adult.csv")
    #new column named kfold
    df["kfold"] = -1
    #randomize the rows of dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    #fetch labels
    y = df.income.values
    #instance of kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)
    #fill new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] =f
    #save new csv as kfolds
    df.to_csv("../input/adult_folds.csv", index=False)