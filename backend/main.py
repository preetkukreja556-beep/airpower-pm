import os, re, datetime as dt
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
import secrets
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import *

model, MU, SD, IMP, AUC = train()
F = make_fleet(); F["prob"] = model.predict_proba(F[FEATS])[:, 1]
F["risk"] = np.where(F.prob >= .6, "HIGH", np.where(F.prob >= .3, "MEDIUM", "LOW"))
rng = np.random.default_rng(3)
uc = set(rng.choice([i for i in F.aircraft_id if i != "A017"], 5, replace=False))
F["status"] = np.where(F.aircraft_id.isin(uc), "UNDER MAINTENANCE", "AVAILABLE")
TYPES = ["TR-1 Trainer", "FJ-2 Fighter", "TP-3 Transport", "HL-4 Helicopter"]  # fictional synthetic types
F["model"] = [TYPES[i % 4] for i in range(len(F))]
F["last_maint"] = [(dt.date.today() - dt.timedelta(days=int(d))).isoformat() for d in F.days_since_last_maintenance]
STARTED = dt.datetime.now()
BAYS, SLOT = 2, 3  # assumption: 2 inspection bays, 3-day slots

ACTION = {"vibration_level": "Engine vibration / bearing & rotor balance inspection",
 "engine_temperature": "Engine hot-section & cooling system inspection", "oil_pressure": "Oil system, pump & filter inspection",
 "hydraulic_pressure": "Hydraulic system leak & pump check", "brake_temperature": "Brake assembly inspection",
 "days_since_last_maintenance": "Overdue scheduled inspection", "component_age": "Life-limited component review",
 "previous_failures": "Repeat-defect root-cause check", "fuel_flow": "Fuel system check", "flight_hours": "Hours-based inspection", "maintenance_count": "Maintenance history review"}
ISSUE = {"vibration_level": "Potential engine/rotating-component degradation", "engine_temperature": "Potential engine thermal stress",
 "oil_pressure": "Potential lubrication system degradation", "hydraulic_pressure": "Potential hydraulic system degradation"}

def factors(row):
    out = []
    for f in FEATS:
        z = (row[f] - MU[f]) / SD[f]; c = IMP[f] * DIR[f] * z
        if DIR[f] * z > .5: out.append((c, f, z))
    out.sort(reverse=True)
    return [{"feature": f, "contribution": round(float(c), 3), "value": float(row[f]), "fleet_mean": round(float(MU[f]), 1),
             "text": ("Low " if DIR[f] < 0 else "High " if f in ("vibration_level","previous_failures","maintenance_count") else "Elevated ") + LABEL[f]}
            for c, f, z in out[:4]]

def top(row):
    fs = factors(row); return fs[0]["feature"] if fs else "flight_hours"

def plan():
    c = F[(F.status == "AVAILABLE") & (F.risk != "LOW")].sort_values("prob", ascending=False)
    t0 = dt.date.today() + dt.timedelta(days=1); out = []
    for k, (_, r) in enumerate(c.iterrows()):
        s = t0 + dt.timedelta(days=(k // BAYS) * SLOT); fs = factors(r)
        out.append({"priority": k+1, "aircraft": r.aircraft_id, "risk": r.risk, "probability": round(r.prob, 3),
                    "inspection": ACTION[top(r)], "window": f"{s} to {s + dt.timedelta(days=SLOT-1)}",
                    "reason": ", ".join(x["text"].lower() for x in fs[:3]) or "elevated model risk"})
    return out

def availability():
    n = len(F); av = F[F.status == "AVAILABLE"]; high = av[av.risk == "HIGH"]
    exp_loss = float(high.prob.sum()); planned = min(BAYS, len(high))
    return {"total": n, "available": len(av), "under_maintenance": n - len(av), "current": round(len(av)/n, 3),
            "high_risk_available": len(high), "expected_unplanned_loss": round(exp_loss, 1),
            "projected_reactive": round(max(len(av) - exp_loss, 0)/n, 3),
            "projected_planned_peak": round((len(av) - planned)/n, 3), "bays": BAYS,
            "at_risk_aircraft": high.sort_values("prob", ascending=False).aircraft_id.tolist(),
            "method": "Expected-value scenario on synthetic data: reactive = available - sum(P(fail) of HIGH-risk aircraft); planned = staggered inspections limited by bay capacity."}

KB = [("Vibration trend", "Rising engine vibration over consecutive flights can indicate bearing wear or rotor imbalance. Recommend borescope and vibration spectrum inspection."),
 ("Engine temperature", "Sustained elevated engine temperature suggests hot-section degradation or cooling inefficiency. Inspect hot section and cooling airflow."),
 ("Oil pressure", "Falling oil pressure may indicate pump wear, filter blockage or leakage. Inspect oil pump, filters and lines."),
 ("Maintenance interval", "Extended time since last maintenance raises latent defect risk. Overdue aircraft should be prioritised for the next available slot."),
 ("Hydraulic system", "Low hydraulic pressure suggests leakage or pump degradation. Perform leak check and pump pressure test."),
 ("Prioritisation policy", "Prioritise by predicted failure probability, then by mission criticality. Schedule within bay capacity to avoid simultaneous unavailability."),
 ("Availability policy", "Planned short inspections are preferred to unplanned groundings because they can be staggered across bays and slots."),
 ("Component age", "Aged components with prior failures need life-limited part review.")]
KBTXT = [f"{t}. {x}" for t, x in KB]
VEC = TfidfVectorizer(stop_words="english").fit(KBTXT); KBM = VEC.transform(KBTXT)

def retrieve(q, k=2):
    s = cosine_similarity(VEC.transform([q]), KBM)[0]; ix = s.argsort()[::-1][:k]
    return [KB[i] for i in ix if s[i] > 0]

COMP = ["Engine", "Hydraulic pump", "Oil filter", "Brake assembly", "Fuel system", "Avionics", "Landing gear", "Hydraulic actuator", "Bearing set"]
TEAMS = ["Line Team A", "Line Team B", "Engine Shop", "Avionics Bay"]
REM = {"Scheduled": "Periodic servicing completed per schedule", "Unscheduled": "Defect reported by crew; rectified and tested", "Inspection": "Routine inspection, no defect found"}
def make_records():
    r = np.random.default_rng(11); out = []
    for k in range(70):
        t = str(r.choice(["Scheduled", "Unscheduled", "Inspection"], p=[.5, .2, .3]))
        out.append({"id": f"WO-{1000+k}", "aircraft": str(F.aircraft_id[int(r.integers(len(F)))]), "date": (dt.date.today() - dt.timedelta(days=int(r.integers(1, 120)))).isoformat(),
                    "component": str(r.choice(COMP)), "type": t, "team": str(r.choice(TEAMS)), "status": "Completed", "remarks": REM[t]})
    for k, a in enumerate(sorted(uc)):
        out.append({"id": f"WO-{1100+k}", "aircraft": a, "date": dt.date.today().isoformat(), "component": str(r.choice(COMP)), "type": "Unscheduled", "team": TEAMS[k % 4], "status": "In progress", "remarks": "Work in progress; awaiting parts / test"})
    return sorted(out, key=lambda x: x["date"], reverse=True)
RECORDS = make_records()

class Q(BaseModel): question: str

app = FastAPI(title="AirPower Predictive Maintenance (SYNTHETIC DEMO)")


# DEMO AUTH ONLY (prototype): hard-coded users, in-memory tokens. Production: SSO/RBAC, hashed credentials, audit logs.
USERS = {"officer": ("air@123", "Maintenance Officer", "Wg Cdr (Demo User)"), "planner": ("fleet@123", "Fleet Planner", "Fleet Planner (Demo User)")}
TOKENS = {}
def auth(authorization: str = Header(None)):
    t = (authorization or "").replace("Bearer ", "")
    if t not in TOKENS: raise HTTPException(401, "Login required")
    return TOKENS[t]

class Login(BaseModel): username: str; password: str

@app.post("/api/login")
def login(b: Login):
    u = USERS.get(b.username.strip().lower())
    if not u or u[0] != b.password: raise HTTPException(401, "Invalid credentials")
    t = secrets.token_hex(16); TOKENS[t] = {"name": u[2], "role": u[1]}
    return {"token": t, **TOKENS[t]}

def row(i):
    r = F[F.aircraft_id == i.upper()]
    if r.empty: raise HTTPException(404, "Unknown aircraft")
    return r.iloc[0]

@app.get("/api/fleet", dependencies=[Depends(auth)])
def fleet():
    a = availability(); rows = []
    for _, r in F.iterrows():
        act = "In maintenance" if r.status != "AVAILABLE" else ACTION[top(r)] if r.risk != "LOW" else "Routine monitoring"
        rows.append({"id": r.aircraft_id, "risk": r.risk, "probability": round(r.prob, 3), "status": r.status, "action": act, "model": r.model, "hours": int(r.flight_hours), "last_maint": r.last_maint})
    rows.sort(key=lambda x: -x["probability"])
    return {"summary": {**a, "high_risk": int((F.risk == "HIGH").sum()), "model_auc": round(AUC, 3)}, "aircraft": rows}

@app.get("/api/aircraft/{aid}", dependencies=[Depends(auth)])
def aircraft(aid: str):
    r = row(aid); v, t = history(r); fs = factors(r)
    return {"id": r.aircraft_id, "risk": r.risk, "probability": round(r.prob, 3), "status": r.status, "factors": fs,
            "issue": ISSUE.get(top(r), "No dominant degradation pattern") if r.risk != "LOW" else "No significant issue predicted",
            "action": ACTION[top(r)] if r.risk != "LOW" else "Routine monitoring",
            "sensors": {f: float(r[f]) for f in FEATS}, "history": {"vibration": v, "temperature": t}}

@app.get("/api/maintenance", dependencies=[Depends(auth)])
def maint(): return plan()

@app.get("/api/availability", dependencies=[Depends(auth)])
def avail(): return availability()

def llm(q, ctx):
    if not os.getenv("ANTHROPIC_API_KEY"): return None
    try:
        import anthropic
        m = anthropic.Anthropic().messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=400,
            system="You are a maintenance copilot. Answer ONLY from CONTEXT (synthetic demo data). The ML model made the prediction; you only explain it. If context lacks the answer, say so.",
            messages=[{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}])
        return m.content[0].text
    except Exception: return None

@app.post("/api/copilot", dependencies=[Depends(auth)])
def copilot(b: Q):
    q = b.question; ids = re.findall(r"A\d{3}", q.upper()); ql = q.lower(); ctx = []
    if ids and ids[0] in set(F.aircraft_id):
        r = row(ids[0]); fs = factors(r)
        ctx.append(f"{r.aircraft_id}: risk {r.risk}, predicted 30-day failure probability {r.prob:.0%} (ML model). Status {r.status}.")
        for x in fs: ctx.append(f"Factor: {x['text']} (value {x['value']}, fleet mean {x['fleet_mean']}).")
        ctx.append(f"Recommended: {ACTION[top(r)]}.")
        kb = retrieve(" ".join(x["text"] for x in fs) + " " + q)
    elif re.search(r"first|priorit|inspect|which aircraft", ql):
        for p in plan()[:3]: ctx.append(f"#{p['priority']} {p['aircraft']} {p['risk']} p={p['probability']:.0%}: {p['inspection']} ({p['window']}). Reason: {p['reason']}.")
        kb = retrieve("prioritisation policy")
    elif re.search(r"availab|fleet|impact", ql):
        a = availability()
        ctx.append(f"Current availability {a['current']:.0%}; {a['high_risk_available']} HIGH-risk aircraft available ({', '.join(a['at_risk_aircraft'])}); expected unplanned loss {a['expected_unplanned_loss']} aircraft; reactive projection {a['projected_reactive']:.0%} vs planned peak {a['projected_planned_peak']:.0%} with {a['bays']} bays.")
        kb = retrieve("availability policy")
    else: kb = retrieve(q, 3)
    ctx += [f"Knowledge [{t}]: {x}" for t, x in kb]
    text = "\n".join(ctx) or "No relevant data or knowledge found."
    ans = llm(q, text) or ("Based on the synthetic fleet data and knowledge base:\n- " + "\n- ".join(ctx) if ctx else text)
    return {"answer": ans, "sources": [t for t, _ in kb], "grounding": ctx}

@app.get("/api/records", dependencies=[Depends(auth)])
def records(): return RECORDS

@app.get("/api/system", dependencies=[Depends(auth)])
def system():
    return {"model": "GradientBoostingClassifier (scikit-learn)", "version": "demo-0.1", "trained_at": STARTED.isoformat(timespec="seconds"), "train_rows": 4000,
            "auc": round(AUC, 3), "data": "Synthetic, seeded generator", "fleet_size": len(F), "kb_entries": len(KB), "work_orders": len(RECORDS),
            "llm": "Enabled" if os.getenv("ANTHROPIC_API_KEY") else "Disabled (template mode)", "environment": "Local prototype", "server_time": dt.datetime.now().isoformat(timespec="seconds")}


app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def index(): return FileResponse("static/index.html")
