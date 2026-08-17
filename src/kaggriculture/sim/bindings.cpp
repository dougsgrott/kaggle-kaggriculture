// pybind11 module (issue 009). The only file in sim/ that knows about Python -- episode.hpp/cpp,
// market.hpp/cpp and sim.hpp/cpp are all plain C++ and stay that way so they're usable from a
// benchmark or future search code without a Python round-trip.
//
// State introspection from Python is intentionally minimal: full field-by-field binding of the
// Tile/PlantTile/AnimalTile hierarchy would be a lot of mechanical code for a path the issue
// itself calls "slow ... for debugging" (PyCallbackPolicy) -- TapePolicy, not a Python callback,
// is the fast path search code (013-015) actually drives. `tile_info()` below returns a plain
// dict with whatever fields are relevant to that tile's kind, which is enough to write a
// debugging policy without exhaustive bindings; extend it if a concrete use needs more.
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <thread>

#include "episode.hpp"

namespace py = pybind11;
using namespace kaggriculture::sim;

namespace {

// Trampoline so a Python class can subclass Policy directly (rarely needed -- prefer
// PyCallbackPolicy for a plain function -- but supported for parity with pybind11 norms).
class PyPolicy : public Policy {
public:
    using Policy::Policy;
    PlayerTurn act(const GameState& state, const MarketTownState& market, int player) override {
        PYBIND11_OVERRIDE_PURE(PlayerTurn, Policy, act, state, market, player);
    }
};

// Wraps a plain Python callable `(state, market, player) -> PlayerTurn`. Explicitly the slow
// path: one Python call per turn per player, GIL-acquiring every time. Fine for debugging a
// handful of episodes; do not put this in a search inner loop.
class PyCallbackPolicy : public Policy {
public:
    explicit PyCallbackPolicy(py::function callback) : callback_(std::move(callback)) {}

    PlayerTurn act(const GameState& state, const MarketTownState& market, int player) override {
        py::gil_scoped_acquire gil;
        py::object result = callback_(state, market, player);
        return result.cast<PlayerTurn>();
    }

private:
    py::function callback_;
};

py::dict tile_info(const GameState& state, int player, int x, int y) {
    if (player < 0 || player >= N_PLAYERS) throw std::out_of_range("player must be 0 or 1");
    if (x < 0 || x >= state.board_size || y < 0 || y >= state.board_size) throw std::out_of_range("tile out of bounds");
    const Tile& t = state.farms[player].tiles[y][x];
    py::dict d;
    switch (t.kind) {
        case TileKind::EMPTY: d["kind"] = "EMPTY"; break;
        case TileKind::LOCKED: d["kind"] = "LOCKED"; break;
        case TileKind::WEED: d["kind"] = "WEED"; break;
        case TileKind::PLANT:
            d["kind"] = "PLANT";
            d["crop"] = static_cast<int>(t.plant.crop);
            d["planted_day"] = t.plant.planted_day;
            d["watered_today"] = t.plant.watered_today;
            d["consecutive_unwatered"] = t.plant.consecutive_unwatered;
            d["yield_units"] = t.plant.yield_units;
            d["max_lifespan_step"] = t.plant.max_lifespan_step;
            d["fertilized_until_day"] = t.plant.fertilized_until_day;
            break;
        case TileKind::COOP:
        case TileKind::PASTURE:
            d["kind"] = (t.kind == TileKind::COOP) ? "COOP" : "PASTURE";
            d["occupied"] = t.animal.occupied;
            if (t.animal.occupied) {
                d["animal"] = static_cast<int>(t.animal.animal);
                d["placed_day"] = t.animal.placed_day;
                d["yield_units"] = t.animal.yield_units;
                d["consecutive_unfed"] = t.animal.consecutive_unfed;
                d["fed_today"] = t.animal.fed_today;
                d["cared_today"] = t.animal.cared_today;
                d["fertilizer_available"] = t.animal.fertilizer_available;
                d["pending_care_bonus"] = t.animal.pending_care_bonus;
            }
            break;
    }
    return d;
}

std::vector<EpisodeResult> run_batch(const std::vector<std::pair<std::shared_ptr<Policy>, std::shared_ptr<Policy>>>& pairs,
                                      const std::vector<EpisodeConfig>& configs, const std::vector<uint64_t>& seeds,
                                      int n_threads) {
    if (pairs.size() != seeds.size() || pairs.size() != configs.size()) {
        throw std::invalid_argument("policy_pairs, configs and seeds must all be the same length");
    }
    size_t n = pairs.size();
    std::vector<EpisodeResult> results(n);
    if (n == 0) return results;

    int threads = std::max(1, std::min(n_threads, static_cast<int>(n)));

    auto worker = [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; i++) {
            results[i] = run_episode(*pairs[i].first, *pairs[i].second, configs[i], seeds[i]);
        }
    };

    {
        // Released for the whole parallel section: pure-C++ policies (TapePolicy) then run with
        // true thread parallelism. A PyCallbackPolicy re-acquires the GIL itself per act() call
        // (see its definition), so mixing policy types here is safe, just not fast for the
        // Python-callback ones.
        py::gil_scoped_release release;
        std::vector<std::thread> workers;
        size_t chunk = (n + threads - 1) / threads;
        for (int t = 0; t < threads; t++) {
            size_t lo = static_cast<size_t>(t) * chunk;
            size_t hi = std::min(n, lo + chunk);
            if (lo >= hi) break;
            workers.emplace_back(worker, lo, hi);
        }
        for (auto& w : workers) w.join();
    }
    return results;
}

}  // namespace

PYBIND11_MODULE(_sim_native, m) {
    m.doc() = "Kaggriculture C++ sim bindings (issue 009): whole-episode and whole-batch driving.";

    py::enum_<Op>(m, "Op")
        .value("PASS", Op::PASS)
        .value("NORTH", Op::NORTH)
        .value("SOUTH", Op::SOUTH)
        .value("EAST", Op::EAST)
        .value("WEST", Op::WEST)
        .value("PICKUP", Op::PICKUP)
        .value("DROP", Op::DROP)
        .value("PLACE", Op::PLACE)
        .value("PLANT", Op::PLANT)
        .value("WATER", Op::WATER)
        .value("HARVEST", Op::HARVEST)
        .value("FERTILIZE", Op::FERTILIZE)
        .value("DIG", Op::DIG)
        .value("BUILD_COOP", Op::BUILD_COOP)
        .value("BUILD_PASTURE", Op::BUILD_PASTURE)
        .value("FEED", Op::FEED)
        .value("COLLECT_FERTILIZER", Op::COLLECT_FERTILIZER)
        .value("CARE", Op::CARE);

    py::enum_<Item>(m, "Item")
        .value("WHEAT", Item::WHEAT)
        .value("CARROT", Item::CARROT)
        .value("TOMATO", Item::TOMATO)
        .value("STRAWBERRY", Item::STRAWBERRY)
        .value("MELON", Item::MELON)
        .value("EGG", Item::EGG)
        .value("MILK", Item::MILK)
        .value("WOOL", Item::WOOL)
        .value("FERTILIZER", Item::FERTILIZER)
        .value("GOOSE", Item::GOOSE)
        .value("COW", Item::COW)
        .value("SHEEP", Item::SHEEP);

    py::enum_<MarketOp>(m, "MarketOp")
        .value("NONE", MarketOp::NONE)
        .value("HIRE", MarketOp::HIRE)
        .value("BUY_LAND", MarketOp::BUY_LAND)
        .value("BUY_SEED", MarketOp::BUY_SEED)
        .value("BUY_PRODUCT", MarketOp::BUY_PRODUCT)
        .value("BUY_ANIMAL", MarketOp::BUY_ANIMAL)
        .value("SELL", MarketOp::SELL);

    py::class_<UnitAction>(m, "UnitAction")
        .def(py::init([](Op op, Item item, int n) { return UnitAction{op, item, n}; }), py::arg("op") = Op::PASS,
             py::arg("item") = Item::WHEAT, py::arg("n") = 1)
        .def_readwrite("op", &UnitAction::op)
        .def_readwrite("item", &UnitAction::item)
        .def_readwrite("n", &UnitAction::n);

    py::class_<MarketOrder>(m, "MarketOrder")
        .def(py::init([](MarketOp op, Item item, int n) { return MarketOrder{op, item, n}; }),
             py::arg("op") = MarketOp::NONE, py::arg("item") = Item::WHEAT, py::arg("n") = 1)
        .def_readwrite("op", &MarketOrder::op)
        .def_readwrite("item", &MarketOrder::item)
        .def_readwrite("n", &MarketOrder::n);

    py::class_<PlayerTurn>(m, "PlayerTurn")
        .def(py::init<>())
        .def_property(
            "unit_actions", [](const PlayerTurn& t) { return std::vector<UnitAction>(t.unit_actions, t.unit_actions + t.n_units_acting); },
            [](PlayerTurn& t, const std::vector<UnitAction>& actions) {
                if (actions.empty() || actions.size() > static_cast<size_t>(MAX_UNITS_PER_PLAYER)) {
                    throw std::invalid_argument("unit_actions must have between 1 and MAX_UNITS_PER_PLAYER entries");
                }
                t.n_units_acting = static_cast<int>(actions.size());
                std::copy(actions.begin(), actions.end(), t.unit_actions);
            })
        .def_property(
            "market_orders", [](const PlayerTurn& t) { return std::vector<MarketOrder>(t.market_orders, t.market_orders + t.n_market_orders); },
            [](PlayerTurn& t, const std::vector<MarketOrder>& orders) {
                if (orders.size() > static_cast<size_t>(MAX_MARKET_ORDERS)) {
                    throw std::invalid_argument("market_orders must have at most MAX_MARKET_ORDERS entries");
                }
                t.n_market_orders = static_cast<int>(orders.size());
                std::copy(orders.begin(), orders.end(), t.market_orders);
            });

    py::class_<MarketConfig>(m, "MarketConfig")
        .def(py::init<>())
        .def_readwrite("max_market_orders_per_turn", &MarketConfig::max_market_orders_per_turn)
        .def_readwrite("farm_hand_cost_mult", &MarketConfig::farm_hand_cost_mult)
        .def_readwrite("weed_spawn_chance", &MarketConfig::weed_spawn_chance)
        .def_readwrite("town_shop_unlock_interval", &MarketConfig::town_shop_unlock_interval)
        .def_readwrite("town_shop_sell_interval", &MarketConfig::town_shop_sell_interval)
        .def_readwrite("town_center_sell_interval", &MarketConfig::town_center_sell_interval);

    py::class_<EpisodeConfig>(m, "EpisodeConfig")
        .def(py::init<>())
        .def_readwrite("episode_steps", &EpisodeConfig::episode_steps)
        .def_readwrite("board_size", &EpisodeConfig::board_size)
        .def_readwrite("turns_per_day", &EpisodeConfig::turns_per_day)
        .def_readwrite("shed_capacity", &EpisodeConfig::shed_capacity)
        .def_readwrite("starting_money", &EpisodeConfig::starting_money)
        .def_readwrite("market", &EpisodeConfig::market);

    py::class_<EpisodeResult>(m, "EpisodeResult")
        .def_readonly("seed", &EpisodeResult::seed)
        .def_readonly("final_money", &EpisodeResult::final_money)
        .def_readonly("daily_money", &EpisodeResult::daily_money)
        .def_readonly("daily_inventory", &EpisodeResult::daily_inventory)
        .def_readonly("shop_draw_order", &EpisodeResult::shop_draw_order)
        .def_readonly("weed_spawn_count", &EpisodeResult::weed_spawn_count);

    py::class_<GameState>(m, "GameState")
        .def(py::init([](int board_size, int turns_per_day, int shed_capacity, double starting_money, uint64_t seed) {
                 return init_game_state(board_size, turns_per_day, shed_capacity, starting_money, seed);
             }),
             py::arg("board_size") = config_defaults::BOARD_SIZE, py::arg("turns_per_day") = config_defaults::TURNS_PER_DAY,
             py::arg("shed_capacity") = config_defaults::SHED_CAPACITY,
             py::arg("starting_money") = config_defaults::STARTING_MONEY, py::arg("seed") = 0)
        .def("copy", [](const GameState& s) { return s; })  // GameState is trivially copyable (see sim.hpp)
        .def_readonly("step", &GameState::step)
        .def_readonly("board_size", &GameState::board_size)
        .def_readonly("turns_per_day", &GameState::turns_per_day)
        .def_readonly("shed_capacity", &GameState::shed_capacity)
        .def_readonly("seed", &GameState::seed)
        .def("farm_money", [](const GameState& s, int p) { return s.farms[p].money; })
        .def("farm_n_units", [](const GameState& s, int p) { return s.farms[p].n_units; })
        .def("farm_unit_pos",
             [](const GameState& s, int p, int u) {
                 Position pos = s.farms[p].unit_pos[u];
                 return py::make_tuple(pos.x, pos.y);
             })
        .def("farm_hires_today", [](const GameState& s, int p) { return s.farms[p].hires_today; })
        .def("private_shed", [](const GameState& s, int p, int item) { return s.privates[p].shed[item]; })
        .def("private_seeds", [](const GameState& s, int p, int crop) { return s.privates[p].seeds[crop]; })
        .def("private_inventory", [](const GameState& s, int p, int u, int item) { return s.privates[p].inventory[u][item]; })
        .def("tile_info", &tile_info);

    py::class_<MarketTownState>(m, "MarketTownState")
        .def(py::init(&init_market_town_state))
        .def("copy", [](const MarketTownState& s) { return s; })
        .def("inventory", [](const MarketTownState& s, int item) { return s.inventory[item]; })
        .def("shop_count", [](const MarketTownState& s, int shop) { return s.shop_count[shop]; })
        .def_readonly("n_shops_unlocked", &MarketTownState::n_shops_unlocked);

    m.def("market_price", &market_price, py::arg("item"), py::arg("inventory"));

    py::class_<Policy, PyPolicy, std::shared_ptr<Policy>>(m, "Policy").def(py::init<>());

    py::class_<TapePolicy, Policy, std::shared_ptr<TapePolicy>>(m, "TapePolicy")
        .def(py::init<std::vector<PlayerTurn>>(), py::arg("tape"))
        .def("__len__", &TapePolicy::size);

    py::class_<PyCallbackPolicy, Policy, std::shared_ptr<PyCallbackPolicy>>(m, "CallbackPolicy")
        .def(py::init<py::function>(), py::arg("callback"));

    m.def("run_episode", &run_episode, py::arg("policy_a"), py::arg("policy_b"), py::arg("config"), py::arg("seed"),
          "Runs one episode to completion and returns an EpisodeResult.", py::call_guard<py::gil_scoped_release>());

    m.def(
        "advance_turns",
        [](GameState& state, MarketTownState& market, Policy& policy_a, Policy& policy_b, int n_turns, const MarketConfig& cfg) {
            advance_turns(state, market, policy_a, policy_b, n_turns, cfg);
        },
        py::arg("state"), py::arg("market"), py::arg("policy_a"), py::arg("policy_b"), py::arg("n_turns"),
        py::arg("market_config") = MarketConfig{},
        "Advances (state, market) by n_turns IN PLACE, with no EpisodeResult bookkeeping -- the "
        "branching-search primitive: GameState.copy()/MarketTownState.copy() a snapshot, then "
        "advance_turns() each candidate branch under a different policy and compare "
        "state.farm_money(player) directly.");

    m.def(
        "run_batch",
        [](const std::vector<std::pair<std::shared_ptr<Policy>, std::shared_ptr<Policy>>>& pairs,
           const std::vector<EpisodeConfig>& configs, const std::vector<uint64_t>& seeds, int n_threads) {
            return run_batch(pairs, configs, seeds, n_threads);
        },
        py::arg("policy_pairs"), py::arg("configs"), py::arg("seeds"), py::arg("n_threads") = 1,
        "Runs many episodes in parallel (releasing the GIL for the pure-C++ case). policy_pairs, "
        "configs and seeds must all be the same length; one episode per (pair, config, seed). "
        "Pass `[config] * len(seeds)` for the common case of one shared config.");
}
