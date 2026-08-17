"""Constants re-exported from the vendored engine — never retyped by hand.

Two sources, both under `vendor/` (issue 002's pinned copy of kaggle-environments 1.32.7):
  - `kaggriculture.py` for the Python-side tables (CROPS, ANIMALS, MARKET_PARAMS, ...) and the
    helper functions (`_fib`, `_hire_cost`, `_shape`) the rest of `model/` is checked against.
  - `kaggriculture.json` for configuration defaults, which the engine reads inline via scattered
    `get(cfg, "key", default)` calls rather than one dict — the JSON schema's `default` fields are
    the single authoritative source for those.

If the engine changes a constant, `vendor/` has to be re-synced (see its header) before this
module picks up the change — it never independently duplicates a value.
"""

import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_DIR = _REPO_ROOT / "vendor"

_spec = importlib.util.spec_from_file_location("_vendored_kaggriculture", _VENDOR_DIR / "kaggriculture.py")
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)

CROPS = _engine.CROPS
ANIMALS = _engine.ANIMALS
PRODUCTS = _engine.PRODUCTS
MARKET_PARAMS = _engine.MARKET_PARAMS
MARKET_I0 = _engine.MARKET_I0
PRICE_FLOOR = _engine.PRICE_FLOOR
HINGE_GAIN = _engine.HINGE_GAIN
SHOPS = _engine.SHOPS
TOWN_CENTER_PRODUCTS = _engine.TOWN_CENTER_PRODUCTS
MAX_SHOP_INSTANCES = _engine.MAX_SHOP_INSTANCES
LAND_ORDER = _engine.LAND_ORDER
LAND_PRICES = _engine.LAND_PRICES
FARM_HAND_COST_MULT = _engine.FARM_HAND_COST_MULT
FARMER_MOVES = _engine.FARMER_MOVES

# Ground-truth reference implementations from the vendored engine (model/ is tested against
# these directly, not just against episode replays).
fib = _engine._fib
hire_cost = _engine._hire_cost
shape = _engine._shape
market_price = _engine.market_price

with open(_VENDOR_DIR / "kaggriculture.json") as _f:
    _spec_json = json.load(_f)

_cfg_schema = _spec_json["configuration"]
CONFIG_DEFAULTS = {
    key: val["default"]
    for key, val in _cfg_schema.items()
    if isinstance(val, dict) and "default" in val
}
CONFIG_DEFAULTS["episodeSteps"] = _cfg_schema["episodeSteps"]
CONFIG_DEFAULTS["actTimeout"] = _cfg_schema["actTimeout"]
