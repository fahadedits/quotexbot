# main.py
"""
TradingView-only backend (FastAPI) for 1M signals with rolling history + pattern confirmation.
- Uses tradingview_ta to get latest price/indicator data.
- Maintains a rolling buffer of candles per pair (approximate OHLC from closes).
- Runs a multi-strategy ensemble + pattern similarity over past windows (100-300).
- Broadcasts signals via WebSocket (/ws?token=...) and serves /candles?pair=...
- Admin endpoints protected by x-admin-key header.

Env vars:
- ADMIN_KEY (string)         -> your admin header value
- SECRET_TOKEN (string)      -> developer token (optional)
- TRADING_PAIRS (csv)        -> e.g. "EUR/USD,GBP/USD,USD/JPY"
- POLL_INTERVAL (seconds)    -> default 1.0
"""

import os
import time
import json
import asyncio
import math
from datetime import datetime, timezone
from typing import Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
from tradingview_ta import TA_Handler, Interval, Exchange
import ta

# ---------- Config / Env ----------
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin-key-change-me")
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "dev-token-123")
PAIRS_CSV = os.getenv("TRADING_PAIRS", "EUR/USD,GBP/USD,USD/JPY,USD/CAD,AUD/USD")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))  # seconds
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "600"))     # keep last n candles (>= 300)
PATTERN_LOOKBACK_MIN = int(os.getenv("PATTERN_MIN", "100"))
PATTERN_LOOKBACK_MAX = int(os.getenv("PATTERN_MAX", "300"))
SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.88"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN", "30"))   # minimal seconds between signals per pair

PAIRS = [p.strip() for p in PAIRS_CSV.split(",") if p.strip()]

# ---------- Storage (simple JSON files) ----------
TOKENS_FILE = "tokens.json"
SIGNALS_FILE = "signals.json"
# create initial files if not present
if not os.path.exists(TOKENS_FILE):
    with open(TOKENS_FILE, "w") as f:
        json.dump({"dev": {"token": SECRET_TOKEN, "label": "dev", "approved": True}}, f, indent=2)
if not os.path.exists(SIGNALS_FILE):
    with open(SIGNALS_FILE, "w") as f:
        json.dump([], f)

def load_tokens():
    try:
        with open(TOKENS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_tokens(toks):
    with open(TOKENS_FILE, "w") as f:
        json.dump(toks, f, indent=2)

def load_signals():
    try:
        with open(SIGNALS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_signals(history):
    with open(SIGNALS_FILE, "w") as f:
        json.dump(history[-5000:], f, indent=2, default=str)

tokens = load_tokens()
signal_history = load_signals()

# ---------- In-memory runtime state ----------
app = FastAPI(title="TradingView Signal Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-pair rolling buffer: list of dicts {time_iso, open, high, low, close}
buffers: Dict[str, List[Dict[str,Any]]] = {pair: [] for pair in PAIRS}
# track last appended minute per pair to avoid duplicates
_last_minute_ts: Dict[str, int] = {pair: 0 for pair in PAIRS}
# last signal time per pair for cooldown
_last_signal_time: Dict[str, float] = {pair: 0.0 for pair in PAIRS}

# WebSocket clients keyed by token -> list of websockets
clients: Dict[str, List[WebSocket]] = {}

# ---------- Utility: TradingView fetch of latest price/indicators ----------
def get_tv_handler_for_pair(pair: str) -> TA_Handler:
    # pair format like "EUR/USD" -> symbol for tradingview_ta: "EURUSD"
    symbol = pair.replace("/", "")
    # use forex screener & FX_IDC exchange (commonly available symbols)
    return TA_Handler(symbol=symbol, screener="forex", exchange="FX_IDC", interval=Interval.INTERVAL_1_MINUTE)

def fetch_latest_from_tradingview(pair: str) -> Dict[str,Any]:
    """
    Return a dict with at least: {'time': iso_str_UTC, 'close': float}
    tradingview_ta provides indicators; we use close from indicators if present.
    """
    handler = get_tv_handler_for_pair(pair)
    analysis = handler.get_analysis()
    ind = analysis.indicators  # dict of indicator values, frequently contains 'close'
    # Try to obtain close value from common keys
    close = None
    for k in ("close", "Close", "last", "LAST", "close_price"):
        if k in ind:
            try:
                close = float(ind[k])
                break
            except Exception:
                pass
    # fallback: try "close" attribute on analysis? else use moving average as proxy
    if close is None:
        # try to parse from "close" in analysis.summary or try to get candles via handler.get_interval
        # as robust fallback, use RSI/other combos to estimate — but we attempt a safer fallback by using moving average:
        try:
            # some builds expose analysis.time
            close = float(analysis.indicators.get(list(analysis.indicators.keys())[-1], 0))
        except Exception:
            close = None
    # timestamp: tradingview_ta doesn't return exact tick time; use UTC now (minute rounded)
    now = datetime.utcnow().replace(second=0, microsecond=0, tzinfo=timezone.utc)
    iso = now.isoformat()
    return {"time": iso, "close": close}

# ---------- Helper: append new candle (approx OHLC) ----------
def append_candle(pair: str, new_close: float, time_iso: str):
    buff = buffers[pair]
    if not buff:
        # no previous candle -> create a candle with open=close
        item = {"time": time_iso, "open": new_close, "high": new_close, "low": new_close, "close": new_close}
        buff.append(item)
        return item
    last = buff[-1]
    last_close = last["close"]
    # create new candle with open = last_close, high/low around closes
    o = float(last_close)
    c = float(new_close)
    h = max(o, c)
    l = min(o, c)
    item = {"time": time_iso, "open": o, "high": h, "low": l, "close": c}
    buff.append(item)
    # maintain buffer size
    if len(buff) > BUFFER_SIZE:
        buff.pop(0)
    return item

# ---------- Indicators & feature engineering ----------
def df_from_buffer(pair: str) -> pd.DataFrame:
    buff = buffers[pair]
    if not buff:
        return pd.DataFrame()
    df = pd.DataFrame(buff)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def compute_indicators_df(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close']
    out = pd.DataFrame(index=df.index)
    out['close'] = close
    out['ema14'] = close.ewm(span=14, adjust=False).mean()
    out['ema50'] = close.ewm(span=50, adjust=False).mean()
    out['ema200'] = close.ewm(span=200, adjust=False).mean()
    # MACD components
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out['macd'] = macd_line
    out['macd_signal'] = macd_line.ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']
    out['rsi'] = ta.momentum.rsi(close, window=14).fillna(50)
    # momentum
    out['roc'] = close.pct_change(1).fillna(0)
    # wick measure approximated by high-low
    out['range'] = (df['high'] - df['low']).fillna(0)
    return out.fillna(0)

def build_feature_vector_window(df_ind: pd.DataFrame, start_idx: int, length: int) -> np.ndarray:
    """
    Build a numeric feature vector for window of candles [start_idx : start_idx+length)
    We'll extract normalized subfeatures per candle: [roc, macd_hist, rsi_norm, range_norm]
    and flatten the window into vector.
    """
    slice_df = df_ind.iloc[start_idx:start_idx+length]
    if slice_df.shape[0] < length:
        # pad with zeros
        pad = pd.DataFrame(0, index=range(length - slice_df.shape[0]), columns=slice_df.columns)
        slice_df = pd.concat([slice_df, pad])
    # normalize features per window
    roc = slice_df['roc'].to_numpy()
    macd = slice_df['macd_hist'].to_numpy()
    rsi = slice_df['rsi'].to_numpy()
    rng = slice_df['range'].to_numpy()
    # scale each to unit vector to reduce scale effects
    def norm(x): 
        x = np.nan_to_num(x)
        n = np.linalg.norm(x)
        return x / (n + 1e-8)
    v = np.concatenate([norm(roc), norm(macd), norm(rsi - 50), norm(rng)])
    return v

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a)==0 or len(b)==0: 
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ---------- Ensemble scoring ----------
def ensemble_score(last_features: dict) -> Dict[str, Any]:
    """
    last_features contains keys: ema14, ema50, ema200, macd_hist, rsi, last_close, range, roc
    We'll compute simple votes and output prob/direction/confidence.
    """
    votes = 0.0
    weights = 1.0
    # trend filter EMA200
    if last_features['ema50'] > last_features['ema200']:
        votes += 1.0 * weights
    else:
        votes -= 1.0 * weights
    # ema14 vs ema50
    if last_features['ema14'] > last_features['ema50']:
        votes += 1.0 * weights
    else:
        votes -= 0.8 * weights
    # macd hist
    if last_features['macd_hist'] > 0:
        votes += 1.0 * weights
    else:
        votes -= 1.0 * weights
    # rsi band
    if 45 <= last_features['rsi'] <= 75:
        votes += 0.6 * weights
    elif last_features['rsi'] < 30 or last_features['rsi'] > 80:
        votes -= 0.8 * weights
    # wick rejection heuristic (small body, long tail near support/resistance) - approximate
    if last_features.get('range',0) > 0 and abs(last_features.get('roc',0)) < 0.005 and last_features.get('range',0) > 0:
        # neutral/small move -> slight negative
        votes -= 0.1
    # raw -> probability via sigmoid
    raw = votes
    prob = 1.0 / (1.0 + math.exp(-raw))
    direction = "UP" if prob > 0.55 else ("DOWN" if prob < 0.45 else "NEUTRAL")
    confidence = int(max(10, min(99, round(abs(prob - 0.5) * 200))))
    return {"prob": prob, "direction": direction, "confidence": confidence, "raw": raw}

# ---------- Pattern confirmation ----------
def pattern_confirmation(pair: str, window_len: int = 8) -> float:
    """
    Compute max similarity of the last `window_len` window to previous windows
    within PATTERN_LOOKBACK_MIN..PATTERN_LOOKBACK_MAX.
    Returns max similarity (0..1).
    """
    df = df_from_buffer(pair)
    if df.empty or len(df) < (PATTERN_LOOKBACK_MIN + window_len + 2):
        return 0.0
    indf = compute_indicators_df(df)
    total = len(indf)
    last_start = total - window_len
    if last_start <= 0:
        return 0.0
    target = build_feature_vector_window(indf, last_start, window_len)
    max_sim = 0.0
    # search previous windows
    start_min = max(0, total - PATTERN_LOOKBACK_MAX - window_len)
    start_max = max(0, total - PATTERN_LOOKBACK_MIN - window_len)
    if start_max <= 0:
        start_max = total - window_len - 1
    for s in range(start_min, start_max):
        v = build_feature_vector_window(indf, s, window_len)
        sim = cosine_similarity(target, v)
        if sim > max_sim:
            max_sim = sim
    return float(max_sim)

# ---------- Decision: check and maybe emit signal ----------
async def evaluate_pair(pair: str):
    """
    Called regularly. Uses latest buffer to decide.
    """
    # need at least some candles
    buff = buffers[pair]
    if len(buff) < 8:
        return None
    df = df_from_buffer(pair)
    indf = compute_indicators_df(df)
    last_row = indf.iloc[-1].to_dict()
    # prepare features for ensemble
    features = {
        "ema14": last_row.get("ema14",0),
        "ema50": last_row.get("ema50",0),
        "ema200": last_row.get("ema200",0),
        "macd_hist": last_row.get("macd_hist",0),
        "rsi": last_row.get("rsi",50),
        "last_close": last_row.get("close",0),
        "range": last_row.get("range",0),
        "roc": last_row.get("roc",0)
    }
    ensemble = ensemble_score(features)
    # pattern confirmation (use window_len 8 by default)
    sim = pattern_confirmation(pair, window_len=8)
    # apply thresholds: require high ensemble prob and pattern sim high
    now_ts = time.time()
    cooldown_ok = (now_ts - _last_signal_time.get(pair, 0.0)) >= COOLDOWN_SECONDS
    will_emit = False
    # dynamic threshold: require prob > 0.80 or (prob > 0.75 and sim > SIMILARITY_THRESHOLD)
    if cooldown_ok and ((ensemble['prob'] >= 0.80) or (ensemble['prob'] >= 0.70 and sim >= SIMILARITY_THRESHOLD)):
        will_emit = True
    if not will_emit:
        return None
    # build signal
    sig = {
        "pair": pair,
        "time": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "direction": ensemble['direction'],
        "confidence": ensemble['confidence'],
        "prob": ensemble['prob'],
        "strategy": "EMA200 trend + EMA14/50 + MACD + RSI + pattern-confirm",
        "sim": sim,
        "last": float(features['last_close'])
    }
    # persist and update last-signal time
    signal_history.append(sig)
    save_signals(signal_history)
    _last_signal_time[pair] = now_ts
    # broadcast to approved tokens
    await broadcast_to_tokens(sig)
    return sig

# ---------- Broadcasting (websockets) ----------
async def broadcast_to_tokens(sig: dict):
    if not clients:
        return
    msg = {"type":"signal","data":sig}
    text = json.dumps(msg)
    # broadcast to every approved token group
    toks = load_tokens()
    for k, meta in toks.items():
        if meta.get("approved"):
            conns = clients.get(meta['token'], [])
            to_remove = []
            for ws in list(conns):
                try:
                    await ws.send_text(text)
                except Exception:
                    to_remove.append(ws)
            for r in to_remove:
                try:
                    clients[meta['token']].remove(r)
                except Exception:
                    pass

# ---------- Background loop: poll TradingView and evaluate ----------
async def background_loop():
    print("Background loop starting. Pairs:", PAIRS, "Poll interval:", POLL_INTERVAL)
    while True:
        try:
            for pair in PAIRS:
                # fetch latest via tradingview_ta
                try:
                    data = fetch_latest_from_tradingview(pair)
                    close = data.get("close")
                    time_iso = data.get("time")
                    if close is None:
                        # skip if close missing
                        continue
                    # determine minute timestamp integer (unix minutes)
                    dt = datetime.fromisoformat(time_iso)
                    minute_ts = int(dt.replace(second=0, microsecond=0).timestamp())
                    # append new candle only if minute changed
                    if minute_ts != _last_minute_ts.get(pair):
                        append_candle(pair, close, dt.isoformat())
                        _last_minute_ts[pair] = minute_ts
                    # We still evaluate every iteration (even within same minute) using last buffered candle + current close
                    # quick evaluate:
                    await evaluate_pair(pair)
                except Exception as e:
                    print("Error fetching/evaluating", pair, e)
                # small inter-pair sleep to avoid hitting rate-limits quickly
                await asyncio.sleep(0.25)
        except Exception as e:
            print("Background loop outer error", e)
        await asyncio.sleep(POLL_INTERVAL)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(background_loop())

# ---------- FastAPI endpoints ----------
@app.get("/candles")
async def get_candles(pair: str = Query(...), count: int = 300):
    if pair not in buffers:
        return JSONResponse({"error":"unknown pair"}, status_code=400)
    buff = buffers[pair][-count:]
    return {"pair": pair, "timeframe":"1M", "candles": buff}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # clients connect with ?token=...
    await websocket.accept()
    q = websocket.query_params
    token = q.get("token")
    if not token:
        await websocket.send_text(json.dumps({"error":"no token provided"}))
        await websocket.close()
        return
    # validate token
    toks = load_tokens()
    allowed = None
    for label, meta in toks.items():
        if meta.get("token") == token and meta.get("approved"):
            allowed = meta
            break
    if not allowed:
        await websocket.send_text(json.dumps({"error":"token not approved"}))
        await websocket.close()
        return
    # register socket
    clients.setdefault(token, []).append(websocket)
    try:
        await websocket.send_text(json.dumps({"type":"welcome","token_label": allowed.get("label","unknown")}))
        while True:
            data = await websocket.receive_text()
            if data.lower() == "ping":
                await websocket.send_text(json.dumps({"type":"pong"}))
    except WebSocketDisconnect:
        try:
            clients[token].remove(websocket)
        except Exception:
            pass

# ---------- Admin endpoints ----------
def require_admin(req: Request):
    key = req.headers.get("x-admin-key")
    if not key or key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

@app.post("/admin/approve_token")
async def admin_approve_token(req: Request):
    require_admin(req)
    body = await req.json()
    label = body.get("label")
    token = body.get("token")
    if not label or not token:
        return JSONResponse({"error":"label & token required"}, status_code=400)
    toks = load_tokens()
    toks[label] = {"token": token, "label": label, "approved": True}
    save_tokens(toks)
    return {"ok": True, "label": label}

@app.post("/admin/revoke_token")
async def admin_revoke_token(req: Request):
    require_admin(req)
    body = await req.json()
    label = body.get("label")
    toks = load_tokens()
    if not label or label not in toks:
        return JSONResponse({"error":"label missing or not exist"}, status_code=400)
    toks[label]["approved"] = False
    save_tokens(toks)
    return {"ok": True}

@app.get("/admin/history")
async def admin_history(req: Request):
    require_admin(req)
    return {"count": len(signal_history), "last": signal_history[-500:]}

@app.post("/admin/backtest")
async def admin_backtest(req: Request):
    require_admin(req)
    body = await req.json()
    pair = body.get("pair")
    lookback = int(body.get("lookback", 300))
    if not pair or pair not in buffers:
        return JSONResponse({"error":"pair required and must be active"}, status_code=400)
    # naive backtest: evaluate how many signals would have matched next candle move (very rough)
    buff = buffers[pair][-lookback:]
    if len(buff) < 10:
        return JSONResponse({"error":"not enough data"}, status_code=400)
    # reconstruct indicator DF
    df = pd.DataFrame(buff)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    indf = compute_indicators_df(df)
    wins = 0; losses = 0; pending = 0; total = 0
    for i in range(8, len(indf)-1):
        # simulate signal at i
        row = indf.iloc[:i+1]
        features = row.iloc[-1].to_dict()
        score = ensemble_score(features)
        sim = pattern_confirmation(pair, window_len=8)
        if ((score['prob'] >= 0.8) or (score['prob'] >= 0.7 and sim >= SIMILARITY_THRESHOLD)):
            # compare next candle
            cur = indf['close'].iloc[i]
            nxt = indf['close'].iloc[i+1]
            move = "UP" if nxt > cur else "DOWN"
            if score['direction'] == move:
                wins += 1
            else:
                losses += 1
            total += 1
        else:
            pending += 1
    winrate = (wins / total * 100) if total else 0.0
    return {"pair":pair, "wins":wins, "losses":losses, "total":total, "winrate":winrate, "pending":pending}

@app.get("/health")
async def health():
    return {"status":"ok", "pairs": PAIRS, "time": datetime.utcnow().isoformat()}

# ---------- Run (when uvicorn main:app is used) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
