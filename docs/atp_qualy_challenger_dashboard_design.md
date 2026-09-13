# ATP QUALY CHALLENGER DASHBOARD
# PHASE 2 FUNCTIONAL DESIGN

# ==========================================================
# GOAL
# ==========================================================

The dashboard becomes the main access point to the platform.

It combines:

- Tennis Analytics
- Betting Analytics
- Match Prediction

for ATP, ATP Qualifying and Challenger.

# ==========================================================
# MAIN MODULES
# ==========================================================

```text
TENNIS ANALYTICS PLATFORM

+-- Home

+-- Player Analysis

+-- Player Betting

+-- Tournament Analysis

+-- Tournament Betting

+-- Match Prediction
```

# ==========================================================
# HOME
# ==========================================================

Purpose:

```text
Executive overview of the platform.
```

Widgets:

```text
Total Matches

Total Players

Total Tournaments

Latest Predictions

Model Accuracy

Model AUC

Backtest Accuracy
```

Current Metrics:

```text
Players:
5,807

Matches:
216,515

Snapshots:
433,030

Model Accuracy:
65.80%

Model AUC:
72.11%

80% Confidence Accuracy:
89.01%
```

# ==========================================================
# PLAYER ANALYSIS
# ==========================================================

Purpose:

```text
Evaluate player performance.
```

Header:

```text
Player Name

ATP Ranking

ATP Points

Age

Height

Hand

Career Elo

Surface Elo
```

Filters:

```text
Surface

All
Hard
Clay
Grass
Indoor
```

```text
Period

Career

Season

Last 10 Matches

Last 20 Matches

Last 50 Matches
```

Sections:

## Overview

```text
Matches

Wins

Losses

Win %

Career Win %

Season Win %
```

## Serve

```text
Aces / Match

Double Faults / Match

1st Serve In %

1st Serve Won %

2nd Serve Won %

Service Games Won %
```

## Return

```text
Break Points Converted %

Return Quality Metrics

Opponent Quality
```

## Pressure

```text
Pressure Index

Break Points Saved %

Break Points Converted %

Tie Break Win %
```

## Opponent Quality

```text
Average Opponent Rank

Median Opponent Rank

Top10 Matches

Top20 Matches

Top50 Matches

Top100 Matches
```

Charts:

```text
Rolling Elo

Rolling Win %

Rolling Serve %

Rolling Pressure Index

Rolling Opponent Rank
```

# ==========================================================
# PLAYER BETTING
# ==========================================================

Source:

```text
Tennis Data Odds
```

Purpose:

```text
Historical profitability analysis.
```

Filters:

```text
Career

Season

Last 2 Years

Last 5 Years
```

```text
Surface

All
Hard
Clay
Grass
Indoor
```

```text
Mode

All Matches

Favourite

Underdog
```

Cards:

## All Matches

```text
ROI

Yield

Wins

Losses

Average Odds
```

## As Favourite

```text
ROI

Yield

Win %

Median Odds
```

## As Underdog

```text
ROI

Yield

Win %

Median Odds
```

Odds Buckets:

```text
Super Favourite

Favourite

Slight Favourite

Coin Flip

Underdog

Big Underdog
```

Chart:

```text
Cumulative Units

+1 Unit Flat Stake
```

Like:

```text
Cobolli Example
```

# ==========================================================
# TOURNAMENT ANALYSIS
# ==========================================================

Purpose:

```text
Tournament profiling.
```

Header:

```text
Tournament

Surface

Level

Years Available
```

Overview:

```text
Matches

Players

Average Rank

Median Rank

Top10 Players

Top20 Players

Top50 Players

Top100 Players
```

Playing Style:

```text
Aces Per Match

Double Faults Per Match

Average Match Duration

Pressure Index

Break Points Per Match
```

Historical Winners:

```text
Champion

Runner Up

Champion Elo

Champion Ranking
```

Charts:

```text
Tournament Strength Over Time

Average Rank Over Time

Elo Distribution
```

# ==========================================================
# TOURNAMENT BETTING
# ==========================================================

Purpose:

```text
Tournament betting tendencies.
```

Sections:

## Favourites

```text
Super Favourite

Favourite

Slight Favourite
```

Metrics:

```text
Matches

Yield

ROI

Average Odds

Win %
```

## Underdogs

```text
Super Dog

Dog

Slight Dog
```

Metrics:

```text
Matches

Yield

ROI

Average Odds

Win %
```

Charts:

```text
Profit By Odds Range

Profit By Surface

Profit By Tournament Year
```

Examples:

```text
Montreal

Indian Wells

Rome

Roland Garros
```

# ==========================================================
# MATCH PREDICTION
# ==========================================================

Purpose:

```text
Predict future matches.
```

Inputs:

```text
Player A

Player B

Surface
```

Autocomplete:

```text
Player Search
```

Main Output:

```text
Player A Win Probability

Player B Win Probability
```

Visualization:

```text
Gauge

Probability Bar

Prediction Card
```

Model Metrics:

```text
Model Probability

Calibrated Probability

Confidence Level
```

Confidence Levels:

```text
50-60%
LOW

60-70%
MEDIUM

70-80%
HIGH

80%+
PREMIUM
```

# ==========================================================
# BETTING VALUE SCREEN
# ==========================================================

Inputs:

```text
Player A Odds

Player B Odds
```

Calculations:

```text
Market Probability

Model Probability

Edge

Expected Value
```

Outputs:

```text
Bet Rating

A+

A

B

C

No Bet
```

Example:

```text
Model Probability:
63%

Market Probability:
55%

Edge:
+8%

Expected Value:
+14.5%
```

# ==========================================================
# TECHNICAL ARCHITECTURE
# ==========================================================

Phase 2 Stack:

```text
Frontend

Streamlit
```

```text
Backend

Python
Pandas
XGBoost
```

```text
Data Layer

Parquet
```

# ==========================================================
# STREAMLIT STRUCTURE
# ==========================================================

```text
streamlit_app/

¦
+-- Home.py
¦
+-- pages/
¦
¦   +-- 01_Player_Analysis.py
¦
¦   +-- 02_Player_Betting.py
¦
¦   +-- 03_Tournament_Analysis.py
¦
¦   +-- 04_Tournament_Betting.py
¦
¦   +-- 05_Match_Prediction.py
¦
+-- data_access/
¦
¦   +-- players.py
¦
¦   +-- tournaments.py
¦
¦   +-- betting.py
¦
¦   +-- predictions.py
¦
+-- assets/
```

# ==========================================================
# PHASE 2 ROADMAP
# ==========================================================

```text
STEP 1

Player Analysis

STEP 2

Tournament Analysis

STEP 3

Player Betting

STEP 4

Tournament Betting

STEP 5

Match Prediction

STEP 6

Dashboard Polish

Dark Theme

Logos

Player Photos

Responsive Layout
```

# ==========================================================
# STATUS
# ==========================================================

```text
PHASE 1
COMPLETED

PHASE 2
READY TO START
```

