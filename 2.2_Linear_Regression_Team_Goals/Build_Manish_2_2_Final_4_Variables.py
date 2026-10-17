import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE.parent / "2.1_Linear_Regression_Goal_Difference"
OUTPUT = HERE / "outputs"
OUTPUT.mkdir(exist_ok=True)

WINDOW = 7
ELO_START = 1500.0
ELO_K = 20.0

results = pd.read_csv(SOURCE / "International_results.csv")
results["date"] = pd.to_datetime(results["date"], format="mixed", dayfirst=True)

age = pd.read_csv(SOURCE / "Age.csv", encoding="cp1252")

name_map = {
    "DR Congo": "Democratic Republic of the Congo",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czech Republic": "Czechia",
    "Turkey": "Turkiye",
}

for c in ["home_team", "away_team"]:
    results[c] = results[c].astype(str).str.strip().replace(name_map)

age["Club"] = age["Club"].astype(str).str.strip().replace(name_map)
results = results.sort_values("date").reset_index(drop=True)

home_hist = results[
    ["date","home_team","away_team","home_score","away_score"]
].rename(columns={
    "home_team":"Team",
    "away_team":"Opponent",
    "home_score":"Goals_For",
    "away_score":"Goals_Against",
})

away_hist = results[
    ["date","away_team","home_team","away_score","home_score"]
].rename(columns={
    "away_team":"Team",
    "home_team":"Opponent",
    "away_score":"Goals_For",
    "home_score":"Goals_Against",
})

history = pd.concat(
    [home_hist, away_hist],
    ignore_index=True
).sort_values(["date","Team"])

wc = results[
    (results["tournament"] == "FIFA World Cup")
    & (results["date"].dt.year == 2026)
].copy().sort_values(["date","home_team","away_team"]).reset_index(drop=True)

if len(wc) != 104:
    raise ValueError(f"Expected 104 FIFA World Cup 2026 matches, found {len(wc)}")

# Age.csv counts the 2026 tournament participation.
# Subtract one so this feature is explicitly pre-2026.
age["Pre2026_WC_Participations"] = age["WC particip."] - 1
wc_part = age.set_index("Club")["Pre2026_WC_Participations"].to_dict()

ratings = {}
elo_pre = {}

for _, r in results.iterrows():
    home = r["home_team"]
    away = r["away_team"]

    home_rating = ratings.get(home, ELO_START)
    away_rating = ratings.get(away, ELO_START)

    key = (
        r["date"], home, away,
        float(r["home_score"]), float(r["away_score"])
    )
    elo_pre[key] = (home_rating, away_rating)

    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
    if r["home_score"] > r["away_score"]:
        score_home = 1.0
    elif r["home_score"] == r["away_score"]:
        score_home = 0.5
    else:
        score_home = 0.0

    ratings[home] = home_rating + ELO_K * (score_home - expected_home)
    ratings[away] = away_rating + ELO_K * ((1-score_home) - (1-expected_home))

rows = []

for i, r in wc.iterrows():
    key = (
        r["date"], r["home_team"], r["away_team"],
        float(r["home_score"]), float(r["away_score"])
    )
    home_elo, away_elo = elo_pre[key]

    for team, opponent, goals, team_elo, opponent_elo in [
        (r["home_team"], r["away_team"], int(r["home_score"]), home_elo, away_elo),
        (r["away_team"], r["home_team"], int(r["away_score"]), away_elo, home_elo),
    ]:
        team_prev = history[
            (history["Team"] == team)
            & (history["date"] < r["date"])
        ].tail(WINDOW)

        opp_prev = history[
            (history["Team"] == opponent)
            & (history["date"] < r["date"])
        ].tail(WINDOW)

        rows.append({
            "Match_ID": i + 1,
            "Date": r["date"].date().isoformat(),
            "Team": team,
            "Opponent": opponent,
            "Goals_Scored": goals,
            "PreMatch_Elo_Difference":
                team_elo - opponent_elo,
            "Pre2026_WC_Participation_Difference":
                wc_part[team] - wc_part[opponent],
            "Opponent_Loss_Rate_Last7":
                (opp_prev["Goals_For"] < opp_prev["Goals_Against"]).mean(),
            "PreMatch_Unbeaten_Rate_Last7":
                (team_prev["Goals_For"] >= team_prev["Goals_Against"]).mean(),
        })

df = pd.DataFrame(rows)

predictors = [
    "PreMatch_Elo_Difference",
    "Pre2026_WC_Participation_Difference",
    "Opponent_Loss_Rate_Last7",
    "PreMatch_Unbeaten_Rate_Last7",
]

if len(df) != 208:
    raise ValueError(f"Expected 208 team-match rows, found {len(df)}")

if df[predictors].isna().any().any():
    raise ValueError("Missing predictor values found.")

print("Rows:", len(df))
print("Unique matches:", df["Match_ID"].nunique())

summary_rows = []

for p in predictors:
    corr = df[p].corr(df["Goals_Scored"])
    z = (df[p] - df[p].mean()) / df[p].std(ddof=1)

    summary_rows.append({
        "Variable": p,
        "Correlation_with_Goals_Scored": corr,
        "Single_Variable_R2": corr**2,
        "Missing_Values": int(df[p].isna().sum()),
        "Z_Score_Abs_Over_3": int((z.abs() > 3).sum()),
    })

    plt.figure(figsize=(8,5))
    plt.scatter(df[p], df["Goals_Scored"], alpha=0.65)
    coef = np.polyfit(df[p], df["Goals_Scored"], 1)
    xx = np.linspace(df[p].min(), df[p].max(), 200)
    plt.plot(xx, coef[0]*xx + coef[1])
    plt.xlabel(p)
    plt.ylabel("Goals Scored")
    plt.title(f"{p} vs Goals Scored (r = {corr:.3f})")
    plt.tight_layout()
    plt.savefig(OUTPUT / f"scatter_{p}.png", dpi=200, bbox_inches="tight")
    plt.close()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT / "manish_final4_summary.csv", index=False)

df[predictors + ["Goals_Scored"]].corr().to_csv(
    OUTPUT / "manish_final4_correlation_matrix.csv"
)

# Check multicollinearity using Variance Inflation Factor (VIF)
def calculate_vif(dataframe, variables):
    vif_rows = []

    for variable in variables:
        y_vif = dataframe[variable].to_numpy(dtype=float)

        other_variables = [v for v in variables if v != variable]
        X_vif = dataframe[other_variables].to_numpy(dtype=float)

        # Add intercept
        X_vif = np.column_stack([np.ones(len(X_vif)), X_vif])

        coefficients, _, _, _ = np.linalg.lstsq(X_vif, y_vif, rcond=None)
        predicted = X_vif @ coefficients

        ss_res = np.sum((y_vif - predicted) ** 2)
        ss_tot = np.sum((y_vif - y_vif.mean()) ** 2)

        r_squared = 1 - (ss_res / ss_tot)
        vif_value = 1 / (1 - r_squared)

        vif_rows.append({
            "Variable": variable,
            "VIF": vif_value
        })

    return pd.DataFrame(vif_rows)


vif = calculate_vif(df, predictors)

vif.to_csv(
    OUTPUT / "manish_final4_vif.csv",
    index=False
)

print("\nVIF:")
print(vif.to_string(index=False))

final_cols = [
    "Match_ID", "Date", "Team", "Opponent", "Goals_Scored"
] + predictors

df[final_cols].to_csv(
    OUTPUT / "Manish_2_2_Final_4_Variables_208_Rows.csv",
    index=False
)

print(summary.to_string(index=False))
print("\nCompleted final pre-match-safe Manish Regression 2.2 four-variable dataset.")
