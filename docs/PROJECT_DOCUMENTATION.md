# Smart Tennis Analytics 2.0

## Roadmap

### Completed

- Player Analysis
- Player Betting
- Tournament Analysis
- Tournament Betting

### Planned

- Match Prediction
- Daily Predictions
- Value Bets

---

# Dashboard Documentation

## Home

Landing page of the application.

### Purpose

Provides navigation to all analytics and betting modules.

### Capabilities

- Access Player Analysis
- Access Player Betting
- Access Tournament Analysis
- Access Tournament Betting
- Future access to Match Prediction
- Future access to Daily Predictions
- Future access to Value Bets

---

# Player Analysis

## Purpose

Analyze ATP player performance using historical match data.

## Main Metrics

### Performance

- Wins
- Losses
- Win Percentage
- Surface Performance
- Tournament Performance

### Ranking

- Current Ranking
- Ranking Evolution
- Historical Performance

### Trends

- Recent Form
- Last 5 Matches
- Last 10 Matches
- Seasonal Performance

## Typical Use Cases

- Compare two players
- Evaluate current form
- Analyze surface specialization
- Identify performance trends

---

# Player Betting

## Purpose

Evaluate player betting performance and profitability.

## Main Metrics

### Betting Results

- Bets
- Wins
- Losses
- ROI
- Profit
- Yield

### Filters

- Surface
- Tournament
- Season
- Odds Range

### Analysis

- Favourite Performance
- Underdog Performance
- High Odds Performance
- Surface Betting Performance

## Typical Use Cases

- Identify profitable player profiles
- Evaluate historical betting strategies
- Compare player profitability

---

# Tournament Analysis

## Purpose

Analyze ATP tournaments from a sporting perspective.

## Main Metrics

### Tournament Statistics

- Matches
- Players
- Champions
- Surface Type

### Historical Analysis

- Tournament History
- Winners
- Finals
- Participation Trends

### Surface Analysis

- Hard
- Clay
- Grass
- Carpet

## Typical Use Cases

- Analyze tournament history
- Evaluate tournament competitiveness
- Review historical champions

---

# Tournament Betting

## Purpose

Evaluate tournament-level betting performance.

## Main Metrics

### Betting Results

- Profit
- ROI
- Yield
- Winning Percentage

### Tournament Breakdown

- Grand Slams
- Masters 1000
- ATP 500
- ATP 250

### Surface Breakdown

- Hard
- Clay
- Grass

## Typical Use Cases

- Detect profitable tournaments
- Evaluate betting opportunities by category
- Compare surface profitability

---

# Data Architecture

## Core Datasets

### Tennis Data

tennis_data_co_uk_v2.parquet

Historical betting and match results.

### TML

tml_v2.parquet

Official ATP match statistics.

### Match Audit

match_link_audit_v2.parquet

Audit dataset for Tennis Data ? TML matching.

### Master Matches

master_matches_v2.parquet

Unified dataset combining Tennis Data and TML.

### Player Match Statistics

player_per_match_statistics_v2.parquet

Player-level match statistics.

### Betting Statistics

player_betting_statistics_atp.parquet

player_betting_statistics_detailed_atp.parquet

tournament_betting_statistics_atp.parquet

---

# Next Phase

## Match Prediction

### Goal

Predict ATP match outcomes before matches are played.

### Planned Features

- Elo Rating
- Surface Elo
- Recent Form
- Head To Head
- Ranking Difference
- Surface Win %
- Tournament History

### Outputs

match_prediction_features_v1.parquet

match_predictions.parquet

---

## Daily Predictions

### Goal

Generate daily ATP match predictions from upcoming matches.

### Outputs

- Daily Match Predictions
- Confidence Score
- Prediction Ranking

---

## Value Bets

### Goal

Identify betting opportunities where model probability exceeds bookmaker implied probability.

### Outputs

- Expected Value
- Edge %
- Recommended Bets

---
