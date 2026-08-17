# No cmake in this environment -- plain make + g++ (13.3.0 verified). See issue 007's
# acceptance criterion: "compiles clean with -O2 -Wall -Wextra".
CXX      ?= g++
CXXSTD   ?= -std=c++20
CXXFLAGS ?= -O2 -Wall -Wextra $(CXXSTD)

.PHONY: all smoke constants clean

all: smoke

# Regenerate constants_generated.hpp from vendor/kaggriculture.py (via kaggriculture.model.constants).
# Run this whenever vendor/ changes; the .hpp is not hand-edited.
constants:
	cd ../../.. && uv run python -m kaggriculture.sim.generate_constants

smoke: sim.cpp sim.hpp smoke_main.cpp constants_generated.hpp
	$(CXX) $(CXXFLAGS) -o smoke sim.cpp smoke_main.cpp
	./smoke

clean:
	rm -f smoke *.o
