# Self-contained entry point: this file becomes submissions/<tag>/main.py verbatim.
# It must not import anything from the `kaggriculture` package or any other repo
# module — /kaggle_simulations/agent/ on Kaggle's grader has none of that installed.
# Replace the body of `agent` as strategy work lands; keep the no-import constraint.


def agent(observation, configuration):
    return {"farmer": ["PASS"], "hands": [], "market": []}
