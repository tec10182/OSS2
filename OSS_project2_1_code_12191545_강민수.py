import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("2019_kbo_for_kaggle_v2.csv")
    print("Problem 1")
    for year in range(2015, 2019):
        print("YEAR:", year)
        df_y = df[df["year"] == year]
        df_h = df_y.sort_values(by="H", ascending=False).head(10)
        df_avg = df_y.sort_values(by="avg", ascending=False).head(10)
        df_hr = df_y.sort_values(by="HR", ascending=False).head(10)
        df_obp = df_y.sort_values(by="OBP", ascending=False).head(10)

        print("안타:", end=" ")
        for name in df_h["batter_name"]:
            print(name, end=" ")
        print("\n타율:", end=" ")
        for name in df_avg["batter_name"]:
            print(name, end=" ")
        print("\n홈런:", end=" ")
        for name in df_hr["batter_name"]:
            print(name, end=" ")
        print("\n출루율:", end=" ")
        for name in df_obp["batter_name"]:
            print(name, end=" ")
        print()

    print("\nProblem 2")

    df_y = df[df["year"] == 2018]
    pos = df_y["cp"].unique()

    for p in pos:
        df_p = df_y[df_y["cp"] == p]
        name = df_p.sort_values(by="war", ascending=False).iloc[0]["batter_name"]
        print(p, ": ", name)

    print("\nProblem 3")
    at = ["R", "H", "HR", "RBI", "SB", "war", "avg", "OBP", "SLG", "salary"]
    cor = df[at].corr()

    ans = cor.drop("salary", axis=1).loc["salary"].idxmax()
    print(ans)
