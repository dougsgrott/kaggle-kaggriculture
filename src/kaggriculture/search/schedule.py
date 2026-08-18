"""Greedy constructive production scheduler (issue 013).

The framing (see issues/013-greedy-scheduler.md): actions are the binding constraint, not land
or money, so the scheduler's job is to spend the 720 available farmer/hand-turns (plus whatever
extra hands buy) on whichever product currently has the best $/action -- *recomputed against a
projected market price*, not the base-price figures `model.economics` reports at face value.

That recomputation matters more than it looks: `model.economics.one_time_crop_profit_per_action`
says MELON is worth ~9x WHEAT at base prices, but MELON is a `TOWN_CENTER_PRODUCTS`-only good (no
shop ever demands it -- `model.regimes`'s own finding) with a season-long town-drain absorption
capacity of only ~30 units (see `_SEASON_ABSORPTION` below), against ~150+ units a single 5x5
quadrant of melon would produce in one cycle. A handful of melon tiles crater its price to the
$1 floor and every further melon action is a *loss* (seed cost exceeds revenue at the floor).
WHEAT/STRAWBERRY/CARROT, despite lower base $/action, are demanded by far more shop types and
sustain ~15x more volume before the same thing happens to them. The greedy ranking below discovers
this on its own (diversifying once a product's marginal $/action falls below the next-best
option's) -- nobody hand-coded "don't monocrop melon".

Simplifications, all deliberate given the "greedy" framing and issue size (see the issue's own
Revision section for the numbers these produce):
  - Market impact is modeled as a single per-product "effective price" driven by *cumulative
    committed production* net of a season-long town-absorption credit (`model.regimes`'s own
    expected-demand closed form: one of each of the 8 shop types, on schedule) -- not a turn-by-
    turn dynamic simulation. This ignores intra-season timing (front-loaded sales look worse in
    reality than this model predicts) and the opponent's own sales; both are conservative
    omissions (they only ever help, by giving the real market more room), not optimistic ones.
  - A tile commits to ONE crop for ONE cycle at a time; when that cycle is harvested the tile goes
    back through the same ranking (so a tile can, in principle, diversify across its own
    replantings as the market shifts under it -- not just across different tiles). `Schedule`
    only stores each tile's *current* role, so search.agent's runtime replanting always plants
    whatever the schedule most recently decided for that position.
  - Animal structures are restricted to the 4 shed-adjacent tiles (one per quadrant) so FEED/CARE
    never needs an extra walk to the shed -- a real cap on herd diversity, but not a binding one:
    even 4 animals fall far short of WOOL/MILK/EGG's season absorption capacity (see numbers).
  - A tile's revenue is credited to the planning ledger at its actual harvest day, one cycle at a
    time (a tile that finishes a cycle goes back through the SAME ranking, rather than being
    permanently pinned to its first crop) -- getting this realistic cost an earlier version of
    this module its money: crediting a tile's *lifetime* revenue at commit time made the plan look
    flush and over-hire/over-plant on day 0, and the resulting cash crunch weeded out everything
    before the first real harvest ever landed. Only animal income stays smoothed to a flat daily
    installment (`total_yield * price / remaining_days`, from build day to season end) rather than
    tracking individual production events -- a much smaller commitment (4 structures, not ~100
    tiles), so the same distortion has nowhere near the same room to compound.
  - Only one-time crops are grown (see CROP_NAMES) -- ongoing crops (TOMATO, STRAWBERRY) pair a
    much longer first_yield_day with a higher seed cost and this module's per-cycle ledger credits
    their whole multi-event run as one lump, which tuning found reliably talks the ranking into a
    capital-black-hole commitment (see CROP_NAMES's own comment for the numbers).

Net result (see the issue's Revision section for the full numbers): a plan that decisively beats
`pass`, `starter` and `random` in `eval.arena`, but does NOT yet decisively beat issue 005's own
baseline -- baseline's simple, fully-committed melon monoculture is a strong, hard-to-beat
opponent specifically because it's the *sole* seller into MELON's tiny absorption pool once this
scheduler (correctly, per the analysis above) declines to compete for it at scale. Closing that
gap is exactly the kind of work issue 014's search is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggriculture.model.constants import ANIMALS, CROPS, MARKET_I0
from kaggriculture.model.economics import (
    animal_yield_and_actions_over_days,
    hand_cost,
    land_unlock_cost,
    minimal_watering_schedule,
    one_time_crop_actions,
)
from kaggriculture.model.price import price as market_price
from kaggriculture.model.regimes import demand_curve_from_shop_counts
from kaggriculture.model.yields import (
    ongoing_crop_production_days,
    ongoing_crop_yield,
    one_time_crop_yield,
)

BOARD_SIZE = 10
HALF = BOARD_SIZE // 2
# Only one-time crops (WHEAT, CARROT, MELON). Ongoing crops (TOMATO, STRAWBERRY) pair a much
# longer first_yield_day (8, 10 vs 2, 2, 10) with a higher seed cost ($50, $100 vs $10, $20, $80),
# and this module's single-"cycle" model credits their ENTIRE multi-event production run as one
# lump at the end of that run -- combined, tuning found this reliably talked the greedy ranking
# into a capital-black-hole commitment: seed cost paid immediately, tied up for the crop's whole
# multi-week run, with nothing coming back until deep into the season, well past the point where
# a small starting budget plus a full crew's daily recurring cost had already gone to zero (see
# the issue's Revision section). A cash-flow-aware ranking that could still use them profitably is
# exactly the kind of improvement issue 014's search is for; excluding them here is a deliberate,
# documented simplification given this issue's "greedy" scope, not an oversight.
CROP_NAMES = [c for c in CROPS if not CROPS[c]["ongoing"]]  # WHEAT, CARROT, MELON
ANIMAL_NAMES = list(ANIMALS)  # GOOSE, COW, SHEEP

# A hand can comfortably tend this many tiles -- both newly planting them AND keeping up with
# ongoing watering/harvesting -- within a 24-turn budget once movement is accounted for (a
# documented heuristic, not derived from an exact routing solve -- see the issue's "movement-aware
# expansion" scope note). Bounds total commitments via the tile-assignment step's active-tile cap
# (`target_crew * TILES_PER_HAND`). Swept 3-12 against 8 seeds (`pass` opponent) after excluding
# the ongoing crops (see CROP_NAMES): 3 gave the best, most robust result (avg ~$15.2k, zero
# failures across 8 seeds); anything above ~5 started re-overcommitting relative to the
# now-smaller (one-time-crop-only) capital budget (see the issue's Revision section).
TILES_PER_HAND = 3

# A hand's daily action budget, for weighing a new hire's recurring fib-priced daily cost against
# what it's actually worth (see build_schedule's crew-size step) -- another documented heuristic,
# not a routing solve: most of a 24-turn day is movement/watering-window idle time, not productive
# PLANT/WATER/HARVEST actions.
ACTIONS_PER_HAND_PER_DAY = 8

# Hard ceiling on the daily crew, independent of land/tile count. fib grows fast enough that
# hiring "as many as land nominally justifies" (up to ~16 for the full 100 tiles) burns the
# ENTIRE day's affordability check on hire cost alone, leaving nothing for seeds/land/animals on
# the very day that matters most (day 0, before any revenue exists) -- see the issue's Revision
# section. Issue 005's baseline hires exactly 8 every day and does fine off it.
MAX_CREW = 8

# Land and animal-structure purchases (one-time costs) require this much money left over
# afterward -- see build_schedule's land step for why: land+animal+a full crew together consume
# the whole $3000 starting budget on day 0, before any crop has had a chance to mature, and there
# is no recovering from $0 (see the issue's Revision section).
CASH_RESERVE = 500.0

# How many brand-new tiles one hand can plant in a single day on top of watering/harvesting its
# existing ones -- caps how fast free_tiles can be claimed. Without this, a day-0 crew of 1 hand
# would "commit" to every tile in the starting quadrant at once (paid for entirely out of
# starting money) with nobody actually able to plant, water, or harvest most of them -- exactly
# the failure this heuristic exists to prevent (see the issue's Revision section).
NEW_PLANTS_PER_HAND_PER_DAY = 3

# Season-long town-absorption capacity per product, assuming issue 012's "expected" shop draw
# (one of each of the 8 shop types, unlocked on schedule) and zero production -- the credit netted
# against our own cumulative committed production before pricing it (see module docstring).
_SEASON_ABSORPTION: dict[str, float] = demand_curve_from_shop_counts([], remaining_slots=8)


def shed_adjacent_tile(quadrant: int) -> tuple[int, int]:
    """The one shed-adjacent tile in `quadrant` (0=NW,1=NE,2=SW,3=SE) -- vendor's
    `_shed_access_tiles`, NWSE order (see sim.cpp's port, cross-checked in tests)."""
    return [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)][quadrant]


def quadrant_tiles(quadrant: int) -> list[tuple[int, int]]:
    """Every (x, y) in `quadrant`, in a boustrophedon (snake) order: left-to-right on even rows,
    right-to-left on odd ones. This is search.agent's whole "movement-aware" contribution -- a
    hand assigned a contiguous run in this order sweeps its block without backtracking, unlike a
    plain raster order (or issue 005 baseline's mod-partition, which visits tiles by list index,
    not by distance)."""
    x0 = 0 if quadrant in (0, 2) else HALF
    y0 = 0 if quadrant in (0, 1) else HALF
    tiles = []
    for row, y in enumerate(range(y0, y0 + HALF)):
        xs = range(x0, x0 + HALF)
        if row % 2 == 1:
            xs = reversed(list(xs))
        tiles.extend((x, y) for x in xs)
    return tiles


def _effective_price(product: str, committed: float) -> int:
    offset = max(0.0, committed - _SEASON_ABSORPTION.get(product, 0.0))
    return market_price(product, MARKET_I0 + offset)


def _crop_cycle(crop: str) -> tuple[int, int, int]:
    """(yield_units, actions, cycle_days) for one full planting-to-harvest(-to-dig) cycle of
    `crop`, minimally watered, unfertilized -- see model.economics/model.yields."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        events = ongoing_crop_production_days(crop)
        return ongoing_crop_yield(crop), 1 + 2 * len(events), events[-1] + 1
    watered = minimal_watering_schedule(crop)
    return one_time_crop_yield(crop, watered), one_time_crop_actions(watered), cd["max_yield_day"] + 1


def _crop_cycle_value(crop: str, day: int, n_days: int, committed: dict[str, float]) -> tuple[float, int, int, int, float]:
    """(ppa, yield_units, actions, cycle_days, revenue) for planting `crop` on a tile that becomes
    free on `day`, for ONE cycle only -- priced at the CURRENT committed level, i.e. what this
    specific cycle earns given everything already committed to `crop` this season, not a
    lifetime average (that would hide exactly the saturation effect this module exists to catch).
    `revenue` is what the planning ledger credits, but only once this cycle is actually harvested
    (`cycle_days` later) -- see module docstring on why lump-crediting a lifetime broke the plan."""
    cd = CROPS[crop]
    if n_days - day < cd["first_yield_day"]:
        return float("-inf"), 0, 0, 0, 0.0
    yield_units, actions, cycle_days = _crop_cycle(crop)
    price = _effective_price(crop, committed[crop])
    revenue = yield_units * price
    ppa = (revenue - cd["seed"]) / actions
    return ppa, yield_units, actions, cycle_days, revenue


def _animal_lifetime_value(animal: str, day: int, n_days: int, committed: dict[str, float]) -> tuple[float, int, int, float]:
    """(ppa, total_yield, total_actions, daily_income) for placing `animal` on day `day`, fed+cared
    every remaining day through season end. `daily_income` smooths the lifetime revenue evenly
    across the remaining days -- a much smaller lump than a crop tile's (4 structures total, not
    ~100 tiles), so unlike crops it's credited as a flat installment rather than tracked per
    production event (see module docstring)."""
    a = ANIMALS[animal]
    remaining = n_days - day
    if remaining < a["first_yield_day"]:
        return float("-inf"), 0, 0, 0.0
    total_yield, total_actions = animal_yield_and_actions_over_days(animal, remaining, care=True)
    price = _effective_price(a["product"], committed[a["product"]])
    ppa = (total_yield * price - a["cost"]) / total_actions
    return ppa, total_yield, total_actions, (total_yield * price) / remaining


@dataclass
class Schedule:
    """The macro plan: what each tile grows/holds for the rest of the season once assigned, plus
    the hire/land calendars. `search.agent.schedule_agent` expands this into a live per-turn
    decision function; nothing here is itself a sequence of primitive actions."""

    tile_role: dict[tuple[int, int], str] = field(default_factory=dict)  # (x,y) -> crop or animal name
    tile_kind: dict[tuple[int, int], str] = field(default_factory=dict)  # (x,y) -> "crop" | "animal"
    tile_first_assigned_day: dict[tuple[int, int], int] = field(default_factory=dict)  # (x,y) -> day of its FIRST commitment
    crew_size_by_day: dict[int, int] = field(default_factory=dict)  # day -> hands to hire THAT morning
    land_days: dict[int, int] = field(default_factory=dict)  # extra-quadrant index (0=NE,1=SW,2=SE) -> day
    n_days: int = 30
    diagnostics: dict = field(default_factory=dict)

    def target_hands(self, day: int) -> int:
        """`_end_of_day` wipes `farm["hands"]` to `[]` every day (vendor's own mechanic -- hands
        are day labour, re-hired from scratch each morning, not a persistent headcount), so this
        is the crew size that should be freshly hired *today*, not a cumulative total."""
        applicable = [d for d in self.crew_size_by_day if d <= day]
        return self.crew_size_by_day[max(applicable)] if applicable else 0


def build_schedule(config: dict, allow_4th_quadrant: bool = True) -> Schedule:
    """Greedy day-by-day construction. Every day: unlock land if affordable, build the
    best-ranked animal structure if a shed-adjacent tile and money allow, hire while the next
    hand's cost (cheap -- it resets daily, see model.economics.hand_cost) still clears the best
    available marginal $/action, then assign every free tile to whichever crop currently ranks
    highest (marginal, price-aware) among the affordable, season-long-fitting options."""
    n_days = config["episodeSteps"] // config["turnsPerDay"]
    hand_cost_mult = config.get("farmHandCostMult", 1)
    max_quadrants = 4 if allow_4th_quadrant else 3

    money = float(config["startingMoney"])
    max_hands_cap = (25 * max_quadrants) // TILES_PER_HAND

    committed: dict[str, float] = {p: 0.0 for p in CROP_NAMES}
    committed.update({ANIMALS[a]["product"]: 0.0 for a in ANIMAL_NAMES})

    tile_role: dict[tuple[int, int], str] = {}
    tile_kind: dict[tuple[int, int], str] = {}
    tile_first_assigned_day: dict[tuple[int, int], int] = {}
    crew_size_by_day: dict[int, int] = {}
    land_days: dict[int, int] = {}
    next_harvest_day: dict[tuple[int, int], int] = {}  # crop tiles only
    pending_revenue: dict[tuple[int, int], float] = {}  # crop tiles only, credited at harvest
    daily_animal_income = 0.0  # sum over all built animal structures, credited every day

    unlocked_quadrants = [0]
    free_tiles: list[tuple[int, int]] = list(quadrant_tiles(0))
    animal_sites = [shed_adjacent_tile(q) for q in range(4)]
    for pos in animal_sites[:1]:
        free_tiles.remove(pos)  # NW's shed-adjacent tile is reserved for the first animal site

    for day in range(n_days):
        # 0. Realized income: already-built animals pay a flat daily installment; crop tiles pay
        # out in full the day their current cycle matures, then free up for reassignment.
        money += daily_animal_income
        for pos in [p for p, hd in next_harvest_day.items() if hd <= day]:
            money += pending_revenue.pop(pos)
            del next_harvest_day[pos]
            free_tiles.append(pos)

        # 1. Land: unlock the next quadrant once affordable WITH a cash reserve left over -- land,
        # an animal structure and a full crew are each individually affordable out of the $3000
        # starting budget, but not all at once on day 0 with nothing left to survive the ~10-day
        # gap before any crop's first harvest (see the issue's Revision section: this was the
        # single most consequential fix found during tuning).
        n_extra = len(unlocked_quadrants) - 1
        if n_extra < max_quadrants - 1:
            cost = land_unlock_cost(n_extra)
            if money - cost >= CASH_RESERVE:
                money -= cost
                quadrant = n_extra + 1
                unlocked_quadrants.append(quadrant)
                land_days[n_extra] = day
                new_tiles = quadrant_tiles(quadrant)
                shed_tile = shed_adjacent_tile(quadrant)
                free_tiles.extend(t for t in new_tiles if t != shed_tile)

        # 2. Animal structures: rank the 3 animal types by current marginal ppa, build the best
        # one at a free shed-adjacent site if it clears the bar and money allows.
        open_sites = [pos for q, pos in enumerate(animal_sites) if q in unlocked_quadrants and pos not in tile_role]
        if open_sites:
            best_animal, best_ppa, best_yield, best_daily_income = None, 0.0, 0, 0.0
            for animal in ANIMAL_NAMES:
                ppa, total_yield, _total_actions, daily_income = _animal_lifetime_value(animal, day, n_days, committed)
                if ppa > best_ppa:
                    best_animal, best_ppa, best_yield, best_daily_income = animal, ppa, total_yield, daily_income
            if best_animal is not None and money - ANIMALS[best_animal]["cost"] >= CASH_RESERVE:
                pos = open_sites[0]
                money -= ANIMALS[best_animal]["cost"]
                daily_animal_income += best_daily_income
                tile_role[pos] = best_animal
                tile_kind[pos] = "animal"
                tile_first_assigned_day.setdefault(pos, day)
                committed[ANIMALS[best_animal]["product"]] += best_yield
                if pos in free_tiles:
                    free_tiles.remove(pos)

        # 3. Crew size: `_end_of_day` wipes `farm["hands"]` every day (vendor's own mechanic), so
        # this is a RECURRING daily rental, not a one-time investment -- fib(0) is cheap again
        # every morning, but so is re-paying fib(1), fib(2), ... for every hand you want again.
        # Grow the daily headcount up to max_hands_cap while the marginal hand's cost still clears
        # a documented per-hand daily action budget, falling back to whatever's actually
        # affordable if the two disagree. NOT gated by tiles already committed: fib(0..7) sums to
        # just 54, trivially affordable from day 0 even before a single tile has been planted --
        # gating crew size on existing commitments (an earlier version of this module did) starves
        # the crew that would go plant those commitments in the first place. Tile assignment
        # (step 4) is what paces commitments to match crew size, not the other way around.
        best_ppa_today = max(
            (max(0.0, _crop_cycle_value(c, day, n_days, committed)[0]) for c in CROP_NAMES),
            default=0.0,
        )
        crew_cap = min(max_hands_cap, MAX_CREW)
        target_crew = 0
        while target_crew < crew_cap and hand_cost(target_crew, hand_cost_mult) < best_ppa_today * ACTIONS_PER_HAND_PER_DAY:
            target_crew += 1
        daily_hire_cost = sum(hand_cost(k, hand_cost_mult) for k in range(target_crew))
        if daily_hire_cost > money:
            target_crew, daily_hire_cost = 0, 0.0
            while target_crew < crew_cap:
                next_cost = daily_hire_cost + hand_cost(target_crew, hand_cost_mult)
                if next_cost > money:
                    break
                daily_hire_cost, target_crew = next_cost, target_crew + 1
        money -= daily_hire_cost
        crew_size_by_day[day] = target_crew

        # 4. Tile assignment: rank crops by current marginal ppa for ONE cycle, assign the best
        # affordable one -- only the seed cost is paid now; the revenue lands (and the tile frees
        # up again) at that cycle's harvest day (step 0, above). A tile already in tile_role is a
        # REPLANT (it just finished a cycle the crew was already tending -- no extra cap); a tile
        # that's never been assigned before is a brand-new commitment, gated two ways: the day's
        # planting throughput (NEW_PLANTS_PER_HAND_PER_DAY) AND a hard ceiling on total active
        # tiles relative to the CURRENT (not some future, larger) crew -- without the second gate,
        # a small early crew keeps accepting new tiles a few at a time forever, so cumulative
        # commitments run ahead of what that crew can actually water/harvest even though each
        # individual day's addition looked affordable.
        new_plant_budget = min(target_crew * NEW_PLANTS_PER_HAND_PER_DAY, max(0, target_crew * TILES_PER_HAND - len(tile_role)))
        for pos in list(free_tiles):
            is_new = pos not in tile_role
            if is_new:
                if new_plant_budget <= 0:
                    continue
                new_plant_budget -= 1
            best_crop, best_ppa, best_yield, best_cycle_days, best_revenue = None, 0.0, 0, 0, 0.0
            for crop in CROP_NAMES:
                if money < CROPS[crop]["seed"]:
                    continue
                ppa, yield_units, _actions, cycle_days, revenue = _crop_cycle_value(crop, day, n_days, committed)
                if ppa > best_ppa:
                    best_crop, best_ppa, best_yield, best_cycle_days, best_revenue = crop, ppa, yield_units, cycle_days, revenue
            if best_crop is None:
                continue
            money -= CROPS[best_crop]["seed"]
            tile_role[pos] = best_crop
            tile_kind[pos] = "crop"
            tile_first_assigned_day.setdefault(pos, day)
            committed[best_crop] += best_yield
            next_harvest_day[pos] = day + best_cycle_days
            pending_revenue[pos] = best_revenue
            free_tiles.remove(pos)

    return Schedule(
        tile_role=tile_role,
        tile_kind=tile_kind,
        tile_first_assigned_day=tile_first_assigned_day,
        crew_size_by_day=crew_size_by_day,
        land_days=land_days,
        n_days=n_days,
        diagnostics={
            "final_money_estimate": money,
            "committed_units": dict(committed),
            "season_absorption": dict(_SEASON_ABSORPTION),
            "final_crew_size": crew_size_by_day.get(n_days - 1, 0),
            "n_crop_tiles": sum(1 for k in tile_kind.values() if k == "crop"),
            "n_animal_tiles": sum(1 for k in tile_kind.values() if k == "animal"),
            "role_counts": {role: sum(1 for r in tile_role.values() if r == role) for role in set(tile_role.values())},
        },
    )
