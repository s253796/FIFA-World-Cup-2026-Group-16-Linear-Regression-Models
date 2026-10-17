MANISH — CORRECTED REGRESSION 2.2 FOUR VARIABLES

Why this correction was made
----------------------------
Two earlier variables could not be safely certified as pre-match:
1. Squad average World Cup matches per player appears to include 2026 appearances.
2. The market-value snapshot date is not documented in the repository.

To remove those assignment-rule risks, both were replaced with predictors that
can be constructed entirely from information available before the match.

Corrected four variables
------------------------
1. PreMatch_Elo_Difference
2. Pre2026_WC_Participation_Difference
3. Opponent_Loss_Rate_Last7
4. PreMatch_Unbeaten_Rate_Last7

Important group note
--------------------
Pre2026_WC_Participation_Difference is also used in Regression 2.1, so it counts
as one shared explanatory variable. The project brief allows up to four variables
to be shared between 2.1 and 2.2.

Results
-------
                           Variable  Correlation_with_Goals_Scored  Missing_Values  Z_Score_Abs_Over_3
            PreMatch_Elo_Difference                         0.4766               0                   0
Pre2026_WC_Participation_Difference                         0.4285               0                   0
           Opponent_Loss_Rate_Last7                         0.2779               0                   0
       PreMatch_Unbeaten_Rate_Last7                         0.2467               0                   0

VIF
---
                           Variable   VIF
            PreMatch_Elo_Difference 4.193
Pre2026_WC_Participation_Difference 2.782
           Opponent_Loss_Rate_Last7 1.502
       PreMatch_Unbeaten_Rate_Last7 1.502

All four corrected predictors have complete 208-row coverage and are based on
pre-match information.
