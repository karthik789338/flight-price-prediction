import json
import os
import joblib
import pandas as pd

from src.preprocessing import engineer_features

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "route_defaults.json")

_model = None
_defaults = None

REQUIRED_RAW = ["city1","city2","quarter","carrier_lg","carrier_low","nsmiles","passengers","large_ms","lf_ms"]

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def load_defaults():
    global _defaults
    if _defaults is None:
        with open(DEFAULTS_PATH, "r") as f:
            _defaults = json.load(f)
    return _defaults

def _parse_body(event):
    # Supports:
    # 1) direct invoke payload dict
    # 2) API Gateway / Function URL style {"body":"{...}"}
    if isinstance(event, dict) and event.get("body") is not None:
        body = event["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return json.loads(body.replace("'", '"'))
        if isinstance(body, dict):
            return body
    return event if isinstance(event, dict) else {}

def _norm(x):
    return str(x).strip() if x is not None else ""

def _lookup_defaults(rec):
    """
    Priority:
      lvl2: city1|city2|quarter
      lvl3: city1|city2
      global
    """
    d = load_defaults()
    c1 = _norm(rec.get("city1"))
    c2 = _norm(rec.get("city2"))
    q  = rec.get("quarter")

    # lvl2 key
    k2 = f"{c1}|{c2}|{q}"
    if c1 and c2 and q is not None and k2 in d.get("lvl2", {}):
        return d["lvl2"][k2]

    # lvl3 key
    k3 = f"{c1}|{c2}"
    if c1 and c2 and k3 in d.get("lvl3", {}):
        return d["lvl3"][k3]

    return d.get("global", {})

def _fill_missing(rec):
    if not isinstance(rec, dict):
        rec = {}

    base = _lookup_defaults(rec)

    # Fill carriers first (your pipeline REQUIRES these raw cols)
    if not rec.get("carrier_lg"):
        rec["carrier_lg"] = base.get("carrier_lg") or load_defaults().get("global", {}).get("carrier_lg")
    if not rec.get("carrier_low"):
        rec["carrier_low"] = base.get("carrier_low") or load_defaults().get("global", {}).get("carrier_low")

    # Fill numeric defaults (if missing)
    for k in ["nsmiles","passengers","large_ms","lf_ms"]:
        if rec.get(k) is None:
            rec[k] = base.get(k)

    # If quarter missing, default to 1 (safe fallback)
    if rec.get("quarter") is None:
        rec["quarter"] = 1

    # Keep city1/city2 as-is; if missing, model will not work anyway
    return rec

def _http_method(event):
    if isinstance(event, dict):
        # Function URL / Lambda Web Adapter style
        rc = event.get("requestContext", {})
        http = rc.get("http", {})
        m = http.get("method")
        if m:
            return m.upper()
        # API Gateway v1 style
        m = event.get("httpMethod")
        if m:
            return str(m).upper()
    return "POST"

def lambda_handler(event, context):
    try:
        method = _http_method(event)

        # Handle browser preflight (CORS)
        if method == "OPTIONS":
            return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

        # GET metadata for UI dropdowns (cities + quarters)
        if method == "GET":
            d = load_defaults()
            # Cities from lvl3 keys (city1|city2)
            cities = set()
            for k in d.get("lvl3", {}).keys():
                a, b = k.split("|")
                cities.add(a)
                cities.add(b)
            out = {"cities": sorted(cities), "quarters": [1,2,3,4]}
            return {"statusCode": 200, "headers": {**CORS_HEADERS, "Content-Type": "application/json"}, "body": json.dumps(out)}

        payload = _parse_body(event)
        records = payload["records"] if isinstance(payload, dict) and "records" in payload else [payload]

        # Fill defaults
        records = [_fill_missing(r) for r in records]

        df = pd.DataFrame(records)

        # Hard safety: ensure required raw columns exist even if empty
        for col in REQUIRED_RAW:
            if col not in df.columns:
                df[col] = None

        df = engineer_features(df)

        model = load_model()
        preds = model.predict(df)

        out = float(preds[0]) if len(preds) == 1 else [float(x) for x in preds]

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"prediction": out})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
