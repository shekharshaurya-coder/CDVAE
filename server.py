"""
server.py — CDVAE Crystal Generator API
All config loaded from config.py (which reads .env)
"""

import traceback
from datetime import datetime, timezone
from typing import Optional

import torch
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import threading
import webbrowser
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


from config import cfg
cfg.validate()

from auth import hash_password, verify_password, create_access_token, get_current_user
from generate import generate, SYMBOL_TO_Z
from src.cdvae_model import (
    CDVAE, NUM_ELEMENTS, MAX_ATOMS,
    params_to_lattice, build_periodic_graph, auto_cutoff
)

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("[WARN] motor not installed — pip install motor")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CDVAE Crystal Generator", version="4.0")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=FileResponse)
async def home():
    return FileResponse("login.html")

@app.get("/index", response_class=FileResponse)
async def index(username: str = Depends(get_current_user)):
    return FileResponse("index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB — URI comes from cfg, never hardcoded ─────────────────────────────
_mongo_client = None

def get_db():
    global _mongo_client
    if not MONGO_AVAILABLE:
        return None
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(cfg.MONGO_URI)
    return _mongo_client[cfg.MONGO_DB]

# ── Load model — checkpoint path from cfg ────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CDVAE(
    hidden=256, latent=128,
    n_enc_layers=4, n_dec_layers=4,
    max_nb=12, n_sigmas=10,
    sigma_begin=0.5, sigma_end=0.005
).to(device)

ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()

# ── PyG graph builder ─────────────────────────────────────────────────────────
class SimpleData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def build_graph_data(frac, atom_types, lat, dev):
    N         = frac.shape[0]
    batch     = torch.zeros(N, dtype=torch.long, device=dev)
    ptr       = torch.tensor([0, N], dtype=torch.long, device=dev)
    n_atoms_t = torch.tensor([N], dtype=torch.long, device=dev)
    lat_b     = lat.unsqueeze(0)
    edge_index, _, edge_dists = build_periodic_graph(
        frac, lat_b, n_atoms_t,
        cutoffs=auto_cutoff(lat_b, n_atoms_t).to(dev),
        max_neighbors=12
    )
    return SimpleData(
        x=atom_types.view(-1, 1).long(), pos=frac,
        batch=batch, ptr=ptr,
        edge_index=edge_index, edge_attr=edge_dists.float(),
        lattice=lat_b,
    )

# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6)
    email:    Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str

class GenerateRequest(BaseModel):
    elements:    list[str]
    n_samples:   int
    temperature: float

class RecomputeRequest(BaseModel):
    cartesian: list[list[float]]
    symbols:   list[str]
    lattice:   list[list[float]]

class SaveStructureRequest(BaseModel):
    structure:   dict
    elements:    list[str]
    temperature: float
    label:       Optional[str] = None

from fastapi.responses import HTMLResponse
import uvicorn

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <h1>CDVAE Login Page</h1>
    """

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES — public
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/register", response_model=TokenResponse, tags=["auth"])
async def register(req: RegisterRequest):
    db = get_db()
    if db is None:
        raise HTTPException(500, "Database not available")
    existing = await db["users"].find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
    await db["users"].insert_one({
        "username":      req.username,
        "email":         req.email,
        "password_hash": hash_password(req.password),
        "created_at":    datetime.now(timezone.utc),
        "last_login":    datetime.now(timezone.utc),
    })
    return TokenResponse(access_token=create_access_token(req.username), username=req.username)


@app.post("/login", response_model=TokenResponse, tags=["auth"])
async def login(req: LoginRequest):
    db = get_db()
    if db is None:
        raise HTTPException(500, "Database not available")
    user = await db["users"].find_one({"username": req.username})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    await db["users"].update_one({"username": req.username}, {"$set": {"last_login": datetime.now(timezone.utc)}})
    return TokenResponse(access_token=create_access_token(req.username), username=req.username)


@app.get("/me", tags=["auth"])
async def me(username: str = Depends(get_current_user)):
    return {"username": username}


# ══════════════════════════════════════════════════════════════════════════════
#  CRYSTAL ROUTES — protected
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate", tags=["crystal"])
def generate_api(req: GenerateRequest, username: str = Depends(get_current_user)):
    try:
        composition = [SYMBOL_TO_Z[e.capitalize()] for e in req.elements]
        results = generate(model, n_samples=req.n_samples, device=device,
                           composition=composition, temperature=req.temperature)
        return results
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/recompute", tags=["crystal"])
@torch.no_grad()
def recompute_api(req: RecomputeRequest, username: str = Depends(get_current_user)):
    try:
        n = len(req.symbols)
        if n < 2:
            return {"error": "Cluster must have at least 2 atoms."}
        try:
            atom_types = torch.tensor([SYMBOL_TO_Z[s.capitalize()] for s in req.symbols],
                                       dtype=torch.long, device=device)
        except KeyError as e:
            return {"error": f"Unknown element symbol: {e}"}
        lat = torch.tensor(req.lattice, dtype=torch.float32, device=device)
        if lat.det().abs().item() < 1e-3:
            return {"error": "Degenerate lattice matrix."}
        cart  = torch.tensor(req.cartesian, dtype=torch.float32, device=device)
        frac  = (cart @ torch.linalg.inv(lat).T) % 1.0
        data  = build_graph_data(frac, atom_types, lat, device)
        mu, _ = model.encoder(data)
        props = model.pred(mu)
        return {
            "energy":   float(props["energy"][0].item()),
            "ehull":    float(max(0.0, props["ehull"][0].item())),
            "band_gap": float(props["bg"][0].item()),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/save", tags=["crystal"])
async def save_structure(req: SaveStructureRequest, username: str = Depends(get_current_user)):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB not available"}
    try:
        result = await db["structures"].insert_one({
            **req.structure,
            "username":           username,
            "requested_elements": req.elements,
            "temperature":        req.temperature,
            "label":              req.label,
            "created_at":         datetime.now(timezone.utc),
        })
        return {"ok": True, "id": str(result.inserted_id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/structures", tags=["crystal"])
async def list_structures(limit: int = 50, skip: int = 0,
                           username: str = Depends(get_current_user)):
    db = get_db()
    if db is None:
        return {"error": "MongoDB not available"}
    try:
        cursor = (
            db["structures"]
            .find({"username": username}, {"_id": 0})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)
    except Exception as e:
        return {"error": str(e)}


@app.delete("/structures/{label}", tags=["crystal"])
async def delete_structure(label: str, username: str = Depends(get_current_user)):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB not available"}
    try:
        result = await db["structures"].delete_one({"username": username, "label": label})
        if result.deleted_count == 0:
            raise HTTPException(404, "Structure not found or not owned by you")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e)}
# server.py  — add this route

from fastapi.responses import PlainTextResponse

@app.get("/structures/{structure_id}/cif", response_class=PlainTextResponse)
async def download_cif(structure_id: str, username: str = Depends(get_current_user)):
    doc = await db.structures.find_one(
        {"_id": ObjectId(structure_id), "username": username}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Structure not found")

    # Build a minimal CIF from stored lattice + fractional coords
    symbols   = doc.get("symbols", [])
    lattice   = doc.get("lattice", [[5,0,0],[0,5,0],[0,0,5]])
    frac      = doc.get("frac_coords", [])   # store frac_coords when saving!
    formula   = "".join(sorted(set(symbols)))
    a_vec, b_vec, c_vec = lattice

    import math
    def vec_len(v): return math.sqrt(sum(x**2 for x in v))
    def vec_dot(u,v): return sum(u[i]*v[i] for i in range(3))

    a = vec_len(a_vec); b = vec_len(b_vec); c = vec_len(c_vec)
    cos_alpha = vec_dot(b_vec,c_vec)/(b*c)
    cos_beta  = vec_dot(a_vec,c_vec)/(a*c)
    cos_gamma = vec_dot(a_vec,b_vec)/(a*b)
    alpha = math.degrees(math.acos(max(-1,min(1,cos_alpha))))
    beta  = math.degrees(math.acos(max(-1,min(1,cos_beta))))
    gamma = math.degrees(math.acos(max(-1,min(1,cos_gamma))))

    lines = [
        "data_cdvae_generated",
        f"_cell_length_a                  {a:.6f}",
        f"_cell_length_b                  {b:.6f}",
        f"_cell_length_c                  {c:.6f}",
        f"_cell_angle_alpha               {alpha:.4f}",
        f"_cell_angle_beta                {beta:.4f}",
        f"_cell_angle_gamma               {gamma:.4f}",
        "_symmetry_space_group_name_H-M  'P 1'",
        "_symmetry_Int_Tables_number      1",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    el_count = {}
    for sym, fc in zip(symbols, frac):
        el_count[sym] = el_count.get(sym, 0) + 1
        label = f"{sym}{el_count[sym]}"
        lines.append(f"  {label:<8} {sym:<4} {fc[0]:>10.6f} {fc[1]:>10.6f} {fc[2]:>10.6f}")

    cif_text = "\n".join(lines) + "\n"
    safe_formula = formula.replace(" ","_")

    from fastapi.responses import Response
    return Response(
        content=cif_text,
        media_type="chemical/x-cif",
        headers={"Content-Disposition": f'attachment; filename="{safe_formula}_cdvae.cif"'}
    )