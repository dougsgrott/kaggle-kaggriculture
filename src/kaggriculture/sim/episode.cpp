#include "episode.hpp"

namespace kaggriculture::sim {

namespace {

void fill_turn_input(const PlayerTurn& a, const PlayerTurn& b, TurnInput& input) {
    const PlayerTurn* turns[N_PLAYERS] = {&a, &b};
    for (int p = 0; p < N_PLAYERS; p++) {
        const PlayerTurn& t = *turns[p];
        input.n_units_acting[p] = t.n_units_acting;
        for (int u = 0; u < t.n_units_acting; u++) input.unit_actions[p][u] = t.unit_actions[u];
        input.n_market_orders[p] = t.n_market_orders;
        for (int o = 0; o < t.n_market_orders; o++) input.market_orders[p][o] = t.market_orders[o];
    }
}

}  // namespace

TurnOutcome run_turn(GameState& state, MarketTownState& market, Policy& policy_a, Policy& policy_b, const MarketConfig& cfg) {
    int day = state.step / state.turns_per_day;

    PlayerTurn a = policy_a.act(state, market, 0);
    PlayerTurn b = policy_b.act(state, market, 1);
    TurnInput input;
    fill_turn_input(a, b, input);

    apply_all_unit_actions(state, input, day);
    process_market_orders(state, market, input.market_orders, input.n_market_orders, cfg);
    town_consume(market, state.step, cfg);

    for (int p = 0; p < N_PLAYERS; p++) decay_plants(state.farms[p], state.step);

    TurnOutcome outcome;
    if ((state.step + 1) % state.turns_per_day == 0) {
        outcome.day_boundary = true;
        for (int p = 0; p < N_PLAYERS; p++) {
            end_of_day_mechanical(state.farms[p], state.privates[p], day, state.board_size, state.turns_per_day,
                                   state.shed_capacity);
            outcome.weed_spawned[p] = spawn_weeds(state, market, p, day, cfg);
        }
        outcome.shop_drawn = unlock_shop(market, day, cfg);
    }

    state.step += 1;
    return outcome;
}

EpisodeResult run_episode(Policy& policy_a, Policy& policy_b, const EpisodeConfig& cfg, uint64_t seed) {
    GameState state = init_game_state(cfg.board_size, cfg.turns_per_day, cfg.shed_capacity, cfg.starting_money, seed);
    MarketTownState market = init_market_town_state();

    EpisodeResult result;
    result.seed = seed;
    // Upper bounds, known up front: avoids reallocation (and the cross-thread allocator
    // contention that comes with it) in run_batch's parallel section.
    int max_days = cfg.episode_steps / cfg.turns_per_day + 1;
    result.daily_money.reserve(max_days);
    result.daily_inventory.reserve(max_days);
    result.shop_draw_order.reserve(MAX_SHOP_INSTANCES);

    for (int step = 0; step < cfg.episode_steps; step++) {
        TurnOutcome outcome = run_turn(state, market, policy_a, policy_b, cfg.market);

        if (outcome.day_boundary) {
            for (int p = 0; p < N_PLAYERS; p++) result.weed_spawn_count[p] += outcome.weed_spawned[p];
            if (outcome.shop_drawn >= 0) result.shop_draw_order.push_back(outcome.shop_drawn);

            std::array<double, N_PLAYERS> money{};
            for (int p = 0; p < N_PLAYERS; p++) money[p] = state.farms[p].money;
            result.daily_money.push_back(money);

            std::array<int, N_PRODUCTS> inv{};
            for (int i = 0; i < N_PRODUCTS; i++) inv[i] = market.inventory[i];
            result.daily_inventory.push_back(inv);
        }
    }

    for (int p = 0; p < N_PLAYERS; p++) result.final_money[p] = state.farms[p].money;
    return result;
}

void advance_turns(GameState& state, MarketTownState& market, Policy& policy_a, Policy& policy_b, int n_turns,
                    const MarketConfig& cfg) {
    for (int i = 0; i < n_turns; i++) {
        run_turn(state, market, policy_a, policy_b, cfg);
    }
}

}  // namespace kaggriculture::sim
