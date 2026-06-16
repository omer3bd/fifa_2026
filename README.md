# FIFA World Cup 2026 - Outcome Prediction using ELO & XGBoost

## **Project Overview**

This project builds a machine learning system that predicts the outcome of international football matches **(Home Win, Draw, or Away Win)** using historical match data, ELO ratings, team form statistics, and head-to-head performance.

The model uses historical international football results to engineer predictive features and trains an XGBoost multiclass classifier to estimate match outcome probabilities.

## **Predicts the probability of**

* Home Team Win
* Draw
* Away Team Win


Example output:
```text
P(Uruguay Win) = 52.4%
P(Draw)        = 24.1%
P(Saudi Arabia Win) = 23.5%

Predicted Outcome:
Uruguay (52.4%)
```

## ** Data Source**
Data is automatically downloaded from the public GitHub repository:

International Football Results Dataset

```https://github.com/martj42/international_results```

| Dataset            | Description                            |
| ------------------ | -------------------------------------- |
| `results.csv`      | Historical international match results |
| `goalscorers.csv`  | Goal scorer information                |
| `shootouts.csv`    | Penalty shootout outcomes              |
| `former_names.csv` | Historical country names               |


# **Data Cleaning pipeline**

* Removes missing scores.
* Converts dates to datetime format.
* Converts scores to integers.
* Encodes neutral venues as binary values.

## **Feature Engineering**

1. **Tournament Weighting**

Different tournaments receive different importance scores.

| Tournament Type         | Weight |
| ----------------------- | ------ |
| FIFA World Cup          | 4      |
| World Cup Qualifiers    | 3      |
| Continental Tournaments | 3      |
| Other Competitions      | 2      |
| Friendly Matches        | 1      |

2. **ELO Rating System**

An ELO rating system is implemented to measure team strength over time.
Generated features:

`home_elo`
`away_elo`
`elo_diff`

Parameters:

```text
Base ELO = 1500
K-Factor = 32
Home Advantage = +60 ELO
```

3. **Team Form Features**

`win5` Average result over the previous 5 matches.

Scoring:
```text
Win  = 1.0
Draw = 0.5
Loss = 0.0
```

`gd5` Average goal difference over the previous 5 matches.

`win10` Average result over the previous 10 matches.

`rest_days` Number of days since the team's previous match.


4. **Head-to-Head Features**

Historical performance between two teams is calculated.

Features:
```
h2h_n → Number of previous meetings
h2h_home_winrate → Historical win rate
h2h_home_gd → Historical average goal difference
```

# How to Use

1. Clone the repository
```bash
https://github.com/omer3bd/fifa_2026.git
```


2. Install dependencies
```bash
pip install pandas numpy requests matplotlib seaborn xgboost scikit-learn
```

3. Configure the match you want to predict

Inside the script, modify these variables:
```python
HOME_TEAM = "Uruguay"

AWAY_TEAM = "Saudi Arabia"

TARGET_DATE = "2026-06-15"

MATCH_NEUTRAL = True

MATCH_WEIGHT = 4
```

| Parameter       | Description                         |
| --------------- | ----------------------------------- |
| `HOME_TEAM`     | Home team name                      |
| `AWAY_TEAM`     | Away team name                      |
| `TARGET_DATE`   | Prediction date (`YYYY-MM-DD`)      |
| `MATCH_NEUTRAL` | `True` if played at a neutral venue |
| `MATCH_WEIGHT`  | Tournament importance (1-4)         |

Tournament weights(MATCH_WEIGHT):
```bash
4 = FIFA World Cup

3 = Qualifiers / Major Continental Tournaments

2 = Regular Competitions

1 = Friendly Matches
```

`MATCH_NEUTRAL` Indicates whether the match is being played at a neutral venue (a stadium that does not belong to either team).
