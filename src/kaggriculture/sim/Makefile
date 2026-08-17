# No cmake in this environment -- plain make + g++ (13.3.0 verified). See issue 007/008's
# acceptance criterion: "compiles clean with -O2 -Wall -Wextra".
CXX      ?= g++
CXXSTD   ?= -std=c++20
CXXFLAGS ?= -O2 -Wall -Wextra $(CXXSTD)

.PHONY: all smoke market-smoke constants clean

all: smoke market-smoke

# Regenerate constants_generated.hpp from vendor/kaggriculture.py (via kaggriculture.model.constants).
# Run this whenever vendor/ changes; the .hpp is not hand-edited.
constants:
	cd ../../.. && uv run python -m kaggriculture.sim.generate_constants

# Issue 007's mechanical-half-only smoke check: no market/town code involved.
smoke: sim.cpp sim.hpp smoke_main.cpp constants_generated.hpp
	$(CXX) $(CXXFLAGS) -o smoke sim.cpp smoke_main.cpp
	./smoke

# Issue 008's economic-half smoke check: price curve, order lockstep, HIRE, town demand.
market-smoke: sim.cpp sim.hpp market.cpp market.hpp pyrandom.cpp pyrandom.hpp market_smoke_main.cpp constants_generated.hpp
	$(CXX) $(CXXFLAGS) -o market_smoke sim.cpp pyrandom.cpp market.cpp market_smoke_main.cpp
	./market_smoke

clean:
	rm -f smoke market_smoke *.o
