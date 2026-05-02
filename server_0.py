from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import traceback
import os
from datetime import datetime, timezone
from typing import Optional

from generate import generate, SYMBOL_TO_Z
from src.cdvae_model import (
    CDVAE, NUM_ELEMENTS, MAX_ATOMS,
    params_to_lattice, build_periodic_graph, auto_cutoff
)
from fastapi.middleware.cors import CORSMiddleware

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("[WARN] motor not installed — run: pip install motor")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://jahagirdaravani:VG4H5JFWnbRcwLYj@cdvae.ew1m0x6.mongodb.net/?appName=cdvae")
MONGO_DB  = os.getenv("MONGO_DB",  "cdvae")
_mongo_client = None

def get_db():
    global _mongo_client
    if not MONGO_AVAILABLE:
        return None
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(MONGO_URI)
    return _mongo_client[MONGO_DB]


# ── Load model once ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CDVAE(
    hidden=256, latent=128,
    n_enc_layers=4, n_dec_layers=4,
    max_nb=12, n_sigmas=10,
    sigma_begin=0.5, sigma_end=0.005
).to(device)

ckpt = torch.load("cdvae_checkpoint_epoch_110.pt", map_location=device)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()


# ── Schemas ───────────────────────────────────────────────────────────────────
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


# ── SimpleData ────────────────────────────────────────────────────────────────
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
        x=atom_types.view(-1,1).long(), pos=frac,
        batch=batch, ptr=ptr,
        edge_index=edge_index, edge_attr=edge_dists.float(),
        lattice=lat_b,
    )


# ── /generate ─────────────────────────────────────────────────────────────────
@app.post("/generate")
def generate_api(req: GenerateRequest):
    try:
        composition = [SYMBOL_TO_Z[e.capitalize()] for e in req.elements]
        results = generate(
            model, n_samples=req.n_samples, device=device,
            composition=composition, temperature=req.temperature
        )
        return results
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── /recompute ────────────────────────────────────────────────────────────────
@app.post("/recompute")
@torch.no_grad()
def recompute_api(req: RecomputeRequest):
    try:
        n = len(req.symbols)
        if n < 2:
            return {"error": "Cluster must have at least 2 atoms."}
        try:
            atom_types = torch.tensor(
                [SYMBOL_TO_Z[s.capitalize()] for s in req.symbols],
                dtype=torch.long, device=device
            )
        except KeyError as e:
            return {"error": f"Unknown element symbol: {e}"}

        lat = torch.tensor(req.lattice, dtype=torch.float32, device=device)
        if lat.det().abs().item() < 1e-3:
            return {"error": "Degenerate lattice matrix (determinant ≈ 0)."}

        cart    = torch.tensor(req.cartesian, dtype=torch.float32, device=device)
        frac    = (cart @ torch.linalg.inv(lat).T) % 1.0
        data    = build_graph_data(frac, atom_types, lat, device)
        mu, _   = model.encoder(data)
        props   = model.pred(mu)

        return {
            "energy":   float(props["energy"][0].item()),
            "ehull":    float(max(0.0, props["ehull"][0].item())),
            "band_gap": float(props["bg"][0].item()),
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── /save  (only called when user explicitly clicks "Save to Atlas") ──────────
@app.post("/save")
async def save_structure(req: SaveStructureRequest):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB not available"}
    try:
        result = await db["structures"].insert_one({
            **req.structure,
            "requested_elements": req.elements,
            "temperature":        req.temperature,
            "label":              req.label,
            "created_at":         datetime.now(timezone.utc),
        })
        return {"ok": True, "id": str(result.inserted_id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── /structures  (list saved structures) ─────────────────────────────────────
@app.get("/structures")
async def list_structures(limit: int = 20, skip: int = 0):
    db = get_db()
    if db is None:
        return {"error": "MongoDB not available"}
    try:
        cursor = db["structures"].find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        return {"error": str(e)}