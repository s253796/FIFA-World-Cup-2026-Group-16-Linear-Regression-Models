MANISH — REGRESSION 2.2 FINAL FOUR VARIABLES

Response variable
-----------------
Goals_Scored

Dataset structure
-----------------
104 FIFA World Cup 2026 matches
208 team-match rows

Final four variables
--------------------
1. Log_Market_Value_Difference
   log(1 + Team Market Value) - log(1 + Opponent Market Value)

2. PreMatch_Elo_Difference
   Team pre-match Elo - Opponent pre-match Elo
   Elo starts at 1500 with K = 20.
   The rating is captured BEFORE the current match is used to update it.

3. WC_Experience_Per_Player_Difference
   Team World Cup matches per player - Opponent World Cup matches per player

4. PreMatch_Unbeaten_Rate_Last7
   Proportion of the team's previous 7 internationals that were wins or draws.

Current evidence
----------------
Log Market Value Difference:
r = 0.5139

Pre-match Elo Difference:
r = 0.4766

WC Experience per Player Difference:
r = 0.4108

Pre-match Unbeaten Rate Last 7:
r = 0.2467

All four variables are available before kickoff and the script does not use
the current match result to construct them.

Important
---------
These are Manish's four variables only.
The final Regression 2.2 model must contain EXACTLY 8 explanatory variables.
Nimeth's two variables and the two agreed shared variables still need to be
merged before the final group model is trained.
