"""SYNTHETIC/DEMO DATA ONLY. Not real aircraft data."""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATS = ["flight_hours","engine_temperature","oil_pressure","vibration_level","fuel_flow",
         "hydraulic_pressure","brake_temperature","component_age","previous_failures",
         "maintenance_count","days_since_last_maintenance"]
LABEL = {"flight_hours":"flight hours","engine_temperature":"engine temperature","oil_pressure":"oil pressure",
         "vibration_level":"vibration level","fuel_flow":"fuel flow","hydraulic_pressure":"hydraulic pressure",
         "brake_temperature":"brake temperature","component_age":"component age",
         "previous_failures":"previous failures","maintenance_count":"maintenance count",
         "days_since_last_maintenance":"days since last maintenance"}
DIR = {f: 1 for f in FEATS}; DIR["oil_pressure"] = -1; DIR["hydraulic_pressure"] = -1

def make_data(n, seed):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "flight_hours": r.uniform(500, 9000, n).round(0),
        "engine_temperature": r.normal(640, 25, n).round(1),
        "oil_pressure": r.normal(55, 4, n).round(1),
        "vibration_level": np.abs(r.normal(2.0, 0.55, n)).round(2),
        "fuel_flow": r.normal(1200, 90, n).round(0),
        "hydraulic_pressure": r.normal(3000, 90, n).round(0),
        "brake_temperature": r.normal(350, 40, n).round(0),
        "component_age": r.uniform(6, 120, n).round(0),
        "previous_failures": r.poisson(1.0, n),
        "maintenance_count": r.poisson(6, n),
        "days_since_last_maintenance": r.uniform(5, 240, n).round(0)})
    z = (d - d.mean()) / d.std()
    logit = (-2.4 + .9*z.vibration_level + .8*z.engine_temperature - .6*z.oil_pressure - .4*z.hydraulic_pressure
             + .5*z.component_age + .4*z.days_since_last_maintenance + .35*z.previous_failures
             + .3*z.brake_temperature + .2*z.fuel_flow + r.normal(0, .5, n))
    d["failure_next_30_days"] = (r.random(n) < 1/(1+np.exp(-logit))).astype(int)
    return d

def train():
    d = make_data(4000, 42)
    Xtr, Xte, ytr, yte = train_test_split(d[FEATS], d.failure_next_30_days, test_size=.25, random_state=1, stratify=d.failure_next_30_days)
    m = GradientBoostingClassifier(random_state=1).fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    return m, d[FEATS].mean(), d[FEATS].std(), dict(zip(FEATS, m.feature_importances_)), auc

def make_fleet(n=40):
    f = make_data(n, 7).drop(columns="failure_next_30_days")
    f.insert(0, "aircraft_id", [f"A{i+1:03d}" for i in range(n)])
    i = 16  # A017: demo high-risk case
    f.loc[i, ["vibration_level","engine_temperature","days_since_last_maintenance","oil_pressure","component_age"]] = [3.6, 700, 210, 49, 96]
    for j, (v, t, dd) in {4: (3.0, 676, 180), 8: (2.9, 668, 195), 22: (3.2, 690, 170), 31: (2.8, 662, 200)}.items():
        f.loc[j, ["vibration_level","engine_temperature","days_since_last_maintenance"]] = [v, t, dd]
    return f

def history(row, days=30):
    r = np.random.default_rng(int(row.aircraft_id[1:]))
    t = np.arange(days)
    vs = max(0, row.vibration_level - 2.0) * .03; ts = max(0, row.engine_temperature - 640) * .02
    vib = row.vibration_level - vs*(days-1-t) + r.normal(0, .04, days)
    tmp = row.engine_temperature - ts*(days-1-t) + r.normal(0, 1.5, days)
    return vib.round(2).tolist(), tmp.round(1).tolist()
