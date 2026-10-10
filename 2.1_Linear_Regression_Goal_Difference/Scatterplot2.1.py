import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# Read dataset into a DataFrame
df = pd.read_csv(HERE / "International_results.csv")
# Read dates as day/month
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)


#Interpreteting match history and sorting data
long = pd.concat([
    df.rename(columns={"home_team": "team", "away_team": "opp", "home_score": "gf", "away_score": "ga"}),
    df.rename(columns={"away_team": "team", "home_team": "opp", "away_score": "gf", "home_score": "ga"})
])[["date", "team", "opp", "gf", "ga"]].sort_values("date")
long["win"] = (long["gf"] > long["ga"]).astype(int)

def form(team, match_date, n=7):
    prev = long[(long["team"] == team) & (long["date"] < match_date)].tail(n)
    if len(prev) == 0:
        return float("nan"), float("nan")
    return prev["win"].mean(), prev["ga"].mean()

# Ensures the goals are registered from FIFA 2026 only
df = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)].copy()
# Calculate goal diff
df["Goal_Diff"] = df["home_score"] - df["away_score"]

#Win Rate and Goals Conceded
win1, win2, conc1, conc2 = [], [], [], []
for _, row in df.iterrows():
    w1, c1 = form(row["home_team"], row["date"])
    w2, c2 = form(row["away_team"], row["date"])
    win1.append(w1); win2.append(w2)
    conc1.append(c1); conc2.append(c2)

df["WinRate_1"], df["WinRate_2"] = win1, win2
df["Conceded_1"], df["Conceded_2"] = conc1, conc2

# Merge in squad stats
age = pd.read_csv(HERE / "Age.csv", encoding="cp1252")
exp = pd.read_csv(HERE / "Experience.csv", encoding="cp1252")

#Caps was exported with a European thousands separator, so 2.930 became 2.93
# while 808 stayed 808. Any value under 100 is really thousands.
exp["Caps"] = exp["Caps"].apply(lambda v: round(v * 1000) if v < 100 else v)


#Template for pulling data
def merge_stat(df, table, col, new_name):
    df = df.merge(table[["Club", col]], left_on="home_team", right_on="Club", how="left") \
            .rename(columns={col: f"{new_name}_1"}).drop(columns="Club")
    df = df.merge(table[["Club", col]], left_on="away_team", right_on="Club", how="left") \
            .rename(columns={col: f"{new_name}_2"}).drop(columns="Club")
    return df

#Template for choosing columns and names [Remove #and change Y1, X1 etc to column selected and new name]
df = merge_stat(df, age, "WC particip.", "WCPart")
df = merge_stat(df, exp, "World Cup matches per player", "WCMatchesPP")

# Sanity check should print 0 for each merged column
for col in ["WCPart_1", "WCPart_2", "WCMatchesPP_1", "WCMatchesPP_2"]:
    print(f"Missing {col}:", df[col].isna().sum())

# TEMPLATE: add one tuple per explanatory variable you want to test 
# (readable label, Home column, Away column)
variables = [
    ("Win Rate (last 7)", "WinRate_1", "WinRate_2"),
    ("Goals Conceded (last 7)", "Conceded_1", "Conceded_2"),
    ("WC Participations", "WCPart_1", "WCPart_2"),
    ("Maximum WC Matches Per Player", "WCMatchesPP_1", "WCMatchesPP_2")
]

#Print scatter
for label, col1, col2 in variables:
    diff = df[col1] - df[col2]

    plt.scatter(diff, df["Goal_Diff"])
    plt.xlabel(f"{label} Difference (Home - Away)")
    plt.ylabel("Goal Difference (Home - Away)")
    plt.title(f"{label} Diff vs Goal Diff")
    plt.show()