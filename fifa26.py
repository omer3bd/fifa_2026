#!/usr/bin/env python
# coding: utf-8

# githib repo: https://github.com/martj42/international_results

# In[1]:


from datetime import datetime
import requests
import pandas as pd
import numpy as np
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report


# In[2]:


def get_data():
    # gibhub raw links.
    former_name_url = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/former_names.csv"
    goalscorers_url = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/goalscorers.csv"
    results_url = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
    shootouts_url = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/shootouts.csv"

    all_data = []

    # loop through github csv datasets.
    print("--- Initiating getting data.")
    for url in [former_name_url, goalscorers_url, results_url, shootouts_url]:

        try:
            print(f"  • getting file: '{url.split('/')[-1]}'.")
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # raises an exception if the download failed

            df = pd.read_csv(StringIO(response.text))
            all_data.append(df)

        except Exception as e:
            print(e)
    print("--- Finished getting data.\n")
    return all_data

# tuple unpacking all datasets that was rereived
former_name_df, goalscorers_df, results_df, shootouts_df = get_data()


# In[3]:


print("--- Initiating with few data adjustments.")

NAME_MAP = {
    # Ireland
    "Éire": "Republic of Ireland",
    "Irish Free State": "Republic of Ireland",

    # Germany
    "German DR": "East Germany",

    # Congo
    "Congo-Kinshasa": "DR Congo",
    "Zaïre": "DR Congo",
    "Belgian Congo": "DR Congo",

    # Yemen
    "Yemen AR": "Yemen",
    "Yemen DPR": "Yemen",

    # Eswatini
    "Swaziland": "Eswatini",

    # Sri Lanka
    "Ceylon": "Sri Lanka",

    # Benin
    "Dahomey": "Benin",

    # Burkina Faso
    "Upper Volta": "Burkina Faso",

    # Tanzania
    "Tanganyika": "Tanzania",

    # Suriname
    "Dutch Guyana": "Suriname",

    # Guyana
    "British Guiana": "Guyana",

    # Zimbabwe
    "Southern Rhodesia": "Zimbabwe",
    "Northern Rhodesia": "Zambia",

    # Ghana
    "Gold Coast": "Ghana",

    # Guinea-Bissau
    "Portuguese Guinea": "Guinea-Bissau",

    # Djibouti
    "French Somaliland": "Djibouti",

    # Malaysia
    "Malaya": "Malaysia",

    # Myanmar
    "Burma": "Myanmar",

    # Vietnam
    "North Vietnam": "Vietnam",
    "Vietnam Republic": "Vietnam",

    # United Arab Republic
    # Historical Egypt-Syria union, usually mapped to Egypt in football datasets
    "United Arab Republic": "Egypt",

    # Curaçao spelling normalization
    "Curaçao": "Curacao",

    # North Macedonia
    "Macedonia": "North Macedonia",

    # Serbia lineage (optional)
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",

    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo"
}

# normalise the texts.
def normalise(name):
    if not isinstance(name, str):
        return name
    return NAME_MAP.get(name, name)

# -----------------------------------------------------------------------
results_df.dropna(subset=["home_score", "away_score"], inplace=True)

results_df["home_team"] = results_df["home_team"].apply(normalise)
results_df["away_team"] = results_df["away_team"].apply(normalise)
results_df["date"] = pd.to_datetime(results_df["date"])

# dropping na and creating a copy.
results = results_df.dropna(subset=["home_score", "away_score"]).copy()

results['home_score'] = results['home_score'].astype(int)
results['away_score'] = results['away_score'].astype(int)
results['neutral'] = results['neutral'].astype(str).str.upper().eq('TRUE').astype(int)


# In[4]:


def team_winner_labels(row):
    if row["home_score"] > row["away_score"]:
        return 0  # home win
    if row["home_score"] < row["away_score"]:
        return 1  # away win
    return 2      # draw

print("\n--- Creating win/lose/draw bool labels for tournaments.")
results['label'] = results.apply(team_winner_labels, axis=1)
print(f"  • labeling process ended.")


# home win(0), away win(1), draw(2)

# In[5]:


# adding weighs to each tournament
big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
def weigh_tournament(name):
    try:
        text = str(name).lower()

        if 'fifa world cup' in text and 'qualif' not in text:
            return 4
        elif 'qualif' in text:
            return 3
        elif any(token in text for token in big):
            return 3
        elif 'friendly' in text:
            return 1
        return 2
    except TypeError:
        print('no strings found.')
    except Exception as e:
        print(e)

print("\n--- Creating weighs for tournaments.")
results['tournament_weight'] = results['tournament'].apply(weigh_tournament)
print(f"  • weighing process ended.")



# ### 1. elo

# In[6]:


ELO_BASE = 1500.0
ELO_K = 32
ELO_HOME_BONUS = 60

# ---------
def compute_elo(results):
    try:
        results = results.sort_values('date').reset_index(drop=True) # reseting and sorting dates in asc order

        rating, home_pre, away_pre  = {}, np.zeros(len(results)), np.zeros(len(results))

        # ---------- elo ----------
        for i, r in results.iterrows():

            # rating home
            rh = rating.get(r.home_team, ELO_BASE)

            # rating away
            ra = rating.get(r.away_team, ELO_BASE)

            home_pre[i] = rh
            away_pre[i] = ra

            # bonus 60 elo if away win, else 0
            bonus = 0 if r.neutral == 1 else ELO_HOME_BONUS

            # new elo formula.
            expected_home = 1 / (1 + 10 ** (-((rh+bonus)-ra)/400))

            score_home = 1.0 if r.label == 0 else (0.5 if r.label == 1 else 0.0)

            margin = abs(int(r.home_score) - int(r.away_score))
            multiplier = np.log(max(margin,1) + 1 ) * (2.2 /(abs(rh - ra)*0.001 + 2.2))

            rating[r.home_team] = rh + ELO_K * multiplier * (score_home - expected_home)
            rating[r.away_team] = ra + ELO_K * multiplier * ((1 - score_home) - (1 - expected_home))

        results['home_elo'], results['away_elo'] = home_pre, away_pre
        results['elo_diff'] = home_pre - away_pre

        print(f"  • ELO calculation complete.")

        return results, rating

    except Exception as e:
        print(f"  • ELO calculation error {e}.")


# In[7]:


def per_team_long(results):

    home = pd.DataFrame({
        "date": results['date'].values,
        'team': results['home_team'].values,
        'opp': results['away_team'].values,
        'gf': results['home_score'].values,
        'ga': results['away_score'].values,
    })

    away = pd.DataFrame({
        "date": results['date'].values,
        'team': results['away_team'].values,
        'opp': results['home_team'].values,
        'gf': results['away_score'].values,
        'ga': results['home_score'].values,
    })

    long = pd.concat([home,away], ignore_index=True)
    long['result'] = np.where(long['gf'] > long['ga'], 1.0, np.where(long['gf'] == long['ga'], 0.5, 0.0))

    long['gd'] = long['gf'] - long['ga']
    print(f"  • added long feature.")

    return long


# In[8]:


def add_form_features(results):
    long = per_team_long(results).sort_values(['team', 'date']).reset_index(drop=True)

    try:
        # ---------- ---------- ---------- ----------
        long['prev_date'] = long.groupby('team')['date'].shift(1)
        long['result_lag'] = long.groupby('team')['result'].shift(1)
        long['gd_lag'] = long.groupby('team')['gd'].shift(1)

        long['win5'] = long.groupby('team')['result_lag'].transform(lambda s: s.rolling(5, min_periods=1).mean())
        long['gd5'] = long.groupby('team')['gd_lag'].transform(lambda s: s.rolling(5, min_periods=1).mean())
        long['win10'] = long.groupby('team')['result_lag'].transform(lambda s: s.rolling(10, min_periods=1).mean())

        long['rest_days'] = (long['date'] - long['prev_date']).dt.days

        form = long[['date', 'team', 'win5', 'gd5', 'win10', 'rest_days']]
        form = form.drop_duplicates(subset=['date', 'team'], keep='first')

        # rename cols for: home team
        results = results.merge(
            form.rename(
                columns={
                    "team": "home_team", 
                    "win5": "home_win5",
                    "gd5": "home_gd5",
                    "win10": "home_win10",
                    "rest_days": "home_rest_days"}
            ),
            on=['date', 'home_team'],
            how="left"

        )

        # rename cols for: away team
        results = results.merge(
            form.rename(
                columns={
                    "team": "away_team", 
                    "win5": "away_win5",
                    "gd5": "away_gd5",
                    "win10": "away_win10",
                    "rest_days": "away_rest_days"}
            ),
            on=['date', 'away_team'],
            how="left"

        )

        print(f"  • Adding complete.")
        return results
    except Exception as e:
        print(f"  • Adding failed {e}.")


# In[9]:


def add_h2h_features(results):
    long = per_team_long(results).sort_values(['team', 'opp', 'date']).reset_index(drop=True)

    grouped = long.groupby(['team', 'opp'])
    long['h2h_n'] = grouped.cumcount()
    long['h2h_winrate'] = grouped['result'].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    long['h2h_gd'] = grouped['gd'].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    h2h = long[['date', 'team', 'opp', 'h2h_n', 'h2h_winrate','h2h_gd']]
    h2h = h2h.drop_duplicates(subset=['date', 'team', 'opp'], keep="first")

    results = results.merge(
        h2h.rename(
            columns={
                'team': 'home_team',
                'opp': 'away_team',
                'h2h_winrate': 'h2h_home_winrate',
                'h2h_gd': 'h2h_home_gd'
            }),
        on=['date', 'home_team', 'away_team'],
        how="left"
    )
    print(f"  • Added h2h.")
    return results


# ### 2. train

# In[10]:


def split_by_date(dataset, train_start, val_start, today_date):

    train = dataset[
        (dataset['date'] >= pd.Timestamp(train_start)) & 
        (dataset['date'] < pd.Timestamp(val_start))
        ].copy()

    val = dataset[
        (dataset['date'] >= pd.Timestamp(val_start)) & 
        (dataset['date'] < pd.Timestamp(today_date))
        ].copy()

    assert train['date'].max() < pd.Timestamp(val_start), "leakage: training row at or after validation cut"
    assert train['date'].max() < pd.Timestamp(today_date), "leakage: validation row at or after today"

    print(f"  • Date split into train and validation.")
    return train, val


# In[11]:


results.columns


# In[12]:


drop = ['date', 'home_team', 'away_team', 'home_score', 'away_score','tournament', 'city', 'country', 'label']
FEATURES = results.columns.difference(drop)

def train_model(train, val):    
    # print("\n--- Creating model.")

    X_train = train[FEATURES].astype(float)
    y_train = train['label'].astype(float) # home win(0), away win(1), draw(2)

    X_val = val[FEATURES].astype(float)
    y_val = val['label'].astype(float) # home win(0), away win(1), draw(2)

    model = XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42
    )

    print(f"  • Train model.")
    model.fit(X_train,y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


# ### 3. evaluate

# In[13]:


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = proba.argmax(axis=1)
    base = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))

    print(f"Validation accuracy: {accuracy_score(y_val, pred):.3f}")
    print(f"Validation log-loss: {log_loss(y_val, proba):.3f} (baseline {log_loss(y_val, base, labels=[0,1,2]):.3f})\n")
    print(classification_report(
        y_val, pred,
        labels=[0, 1, 2],
        target_names=['home_win', 'draw', 'away_win'],
        digits=3,
        zero_division=0
    ))


# ### 4. feature importance

# In[14]:


def print_feature_importance(model, top_n=10):
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Top features by importance:")
    for name, value in importance.head(top_n).items():
        print(f"   {name:<20}  {value:.3f}")


# In[15]:


def form_as_of(long, team, asof_date):
    sub = long[(long['team'] == team) & (long['date'] < pd.Timestamp(asof_date))].sort_values('date')

    if len(sub) == 0:
        return {
            'win5': 0.5, 
            'gd5': 0.0, 
            'win10': 0.5, 
            'rest_days':30.0
        }

    last5, last10 = sub.tail(5), sub.tail(10)

    return {
        'win5': float( last5['result'].mean() ),
        'gd5': float((last5['gf'] - last5['ga']).mean()),
        'win10': float(last10['result'].mean()),
        'rest_days': float( (pd.Timestamp(asof_date) - sub['date'].max()).days ),
    }

def h2h_as_of(long, team, opp, asof_date):
    sub = long[ 
        (long['team'] == team) & 
        (long['opp'] == opp ) & 
        (long['date'] < pd.Timestamp(asof_date))
        ]
    if len(sub) == 0:
        return 0.0, np.nan, np.nan
    return float( len(sub)), float(sub['result'].mean()), float(sub['gd'].mean())

def build_match_row(long, final_elo, home, away, neutral, weight, asof_date):
    home_form = form_as_of(long, home, asof_date)
    away_form = form_as_of(long, away, asof_date)

    home_elo = final_elo.get(home, ELO_BASE)
    away_elo = final_elo.get(away, ELO_BASE)

    h2h_n, h2h_wr, h2h_gd = h2h_as_of(long, home, away, asof_date)

    # ----------------
    row = {
        "neutral": int(neutral),
        "tournament_weight": weight,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "home_wins": home_form["win5"],
        "away_wins": away_form["win5"],
        "home_gd5": home_form["gd5"],
        "away_gd5": away_form["gd5"],
        "home_win10": home_form["win10"],
        "away_win10": away_form["win10"],
        "home_rest_days": home_form["rest_days"],
        "away_rest_days": away_form["rest_days"],
        "h2h_n": h2h_n,
        "h2h_home_winrate": h2h_wr,
        "h2h_home_gd": h2h_gd
    }

    return pd.DataFrame([row])[FEATURES].astype(float)


# In[16]:


# # ── fixtures ──────────────────────────────────────────────────────────────────
# def map_fixture_name(name):
#     name = name.strip()
#     return FIXTURE_NAME_MAP.get(name, name)

# def _side_matches(user_input, raw_name):
#     """True if the user's typed team matches a fixture side (by raw or mapped name)."""
#     u = user_input.strip().lower()
#     return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


# In[17]:


# CACHE_DIR = "data_cache"
# FIXTURES_PATH = os.path.join(CACHE_DIR, "fixtures.csv")

# # maps fixtures.csv team names -> the normalized results.csv names
# FIXTURE_NAME_MAP = {
#     "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
#     "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
#     "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
#     "Cape Verde": "Cabo Verde",
# }


# def find_fixture(team_a, team_b):
#     """Find the single fixture for the two named teams (order doesn't matter)."""
#     fx = pd.read_csv(FIXTURES_PATH)
#     for _, row in fx.iterrows():
#         if " v " not in str(row["teams"]):
#             continue
#         left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
#         forward = _side_matches(team_a, left) and _side_matches(team_b, right)
#         reverse = _side_matches(team_a, right) and _side_matches(team_b, left)
#         if forward or reverse:
#             return {"match": row.get("match_number", ""), "group": row.get("group", ""),
#                     "stadium": row.get("stadium", ""), "date": row.get("date_dt", ""),
#                     "home_disp": left, "away_disp": right,
#                     "home": map_fixture_name(left), "away": map_fixture_name(right)}
#     return None


# In[18]:


# def list_team_names():
#     fx = pd.read_csv(FIXTURES_PATH)
#     names = set()
#     for t in fx["teams"]:
#         if " v " in str(t):
#             for p in str(t).split(" v "):
#                 p = p.strip()
#                 if not any(w in p.lower() for w in ["winner", "runner", "third", "place", "group"]):
#                     names.add(p)
#     return sorted(names)


# In[19]:


# def tag_match(top_prob, p_home, p_away, home_elo, away_elo):
#     favorite_is_home = p_home >= p_away
#     fav_elo_is_home = home_elo >= away_elo
#     upset = (favorite_is_home != fav_elo_is_home)
#     if top_prob >= 0.60:
#         strength = "LOCK"
#     elif top_prob >= 0.45:
#         strength = "LEAN"
#     else:
#         strength = "TOSS-UP"
#     return strength + ("  ⚠️ UPSET PICK" if upset else "")


# ### 5. predict

# In[20]:


def predict_symmetric(model, long, final_elo, team_a, team_b, asof_date, neutral, weight):
    x_ab = build_match_row(long, final_elo, team_a, team_b, neutral, weight, asof_date)
    x_ba = build_match_row(long, final_elo, team_b, team_a,neutral, weight, asof_date)

    p_ab = model.predict_proba(x_ab)[0]
    p_ba = model.predict_proba(x_ba)[0]

    p_a_win = (p_ab[0] + p_ba[2])/ 2.0
    p_draw = (p_ab[1] + p_ba[1])/ 2.0
    p_b_win = (p_ab[2] + p_ba[0])/ 2.0

    total = p_a_win + p_draw + p_b_win
    return p_a_win/ total, p_draw / total, p_b_win / total


# In[21]:


results[results['home_team'] == 'Uruguay'].head(1)


# In[23]:


TRAIN_START = "2006-01-01"
VAL_START = "2023-01-01"
TODAY = "2026-06-15"
TARGET_DATE = "2026-06-15"

HOME_TEAM = "Uruguay"
AWAY_TEAM = "Saudi Arabia"
MATCH_NEUTRAL = True
MATCH_WEIGHT = 4
'''
Sweden	Tunisia
Saudi Arabia
'''

# -------------------------------
print("\n--- Calculating team ELO. [1]")
results, final_elo = compute_elo(results)
print(f'  • {len(results)} total matches')

print("\n--- Adding form features.")
results = add_form_features(results)

print("\n--- Creating h2h features.")
results = add_h2h_features(results)

print("\n--- Splitting date by time.")
train, val = split_by_date(dataset=results, train_start=TRAIN_START, val_start=VAL_START, today_date=TODAY)

print("\n--- Training model. [2]")
model, X_val, y_val = train_model(train, val)

print("\n--- Evaluating model. [3]")
evaluate(model, X_val, y_val)

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("  Top features")
for name, value in importance.head(5).items():
    print(f'   {name:<20} {value:.3f}')
print()

print("\n--- Predicting model. [3]")
long = per_team_long(results)

bra_elo = final_elo.get(HOME_TEAM, ELO_BASE)
mor_elo = final_elo.get(AWAY_TEAM, ELO_BASE)
print(f'\n   - Elo ratings - \n   {HOME_TEAM}: {bra_elo:.0f} | {AWAY_TEAM}: {mor_elo:.0f}\n')

bra_form = form_as_of(long, HOME_TEAM, TARGET_DATE)
mor_form = form_as_of(long, AWAY_TEAM, TARGET_DATE)
print(f"   Last 5 win rate - {HOME_TEAM}: {bra_form['win5']} | {AWAY_TEAM}: {mor_form['win5']}")

print(f"   Last 5 goal diff - {HOME_TEAM}: {bra_form['gd5']} | {AWAY_TEAM}: {mor_form['gd5']}")

# predict
p_home, p_draw, p_away = predict_symmetric(
    model, long, final_elo, HOME_TEAM, AWAY_TEAM, TARGET_DATE, 
    neutral=MATCH_NEUTRAL, weight=MATCH_WEIGHT)

print('-'*62)
print(f"   P({HOME_TEAM}) win = {p_home:.1%}")
print(f"   P(Draw) win = {p_draw:.1%}")
print(f"   P({AWAY_TEAM}) win = {p_away:.1%}")
print('-'*62)

outcomes = (
    (HOME_TEAM, p_home), 
    ("Draw", p_draw), 
    (AWAY_TEAM, p_away)
)

winner, prob = max(outcomes, key=lambda pair: pair[1])
print(f"   >>> Predicted outcome: {winner} ({prob:.1%})")


# In[ ]:


# print_feature_importance(model=model)

# team = 'Brazil'
# opp = 'Morocco'
# asof_date='2026-06-14'

# print("---")
# print(f"{team} vs {opp} - {asof_date}")
# # -------------------------------
# long = per_team_long(results)

# form = form_as_of(
#     long=long, 
#     team=team,
#     asof_date=asof_date
# )
# print(form)
# # -------------------------------




# # -------------------------------
# h2h_ = h2h_as_of(long=long,
#          team=team,
#          opp=opp,
#          asof_date=asof_date)
# print(h2h_)
# # -------------------------------


# In[ ]:


df = results_df[results_df['date'] > '2026-06-12']
df


# In[ ]:


# df.to_csv('fifa_live26.csv', index=False)


# ### output description

# `win5=0.7` Over the last 5 matches, Brazil averaged 0.7 points.
# * Win = 1.0
# * Draw = 0.5
# * Loss = 0.0
# 
# `gd5=1.0` Average goal difference over the last 5 matches.
# 
# `win10=0.75` Over the last 10 matches.
# 
# `rest_days=6` 6 days since last match

# In[ ]:




