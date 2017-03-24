# A GNU make file for compiling the example programme

# Compiler (g++)
CPP:=g++
CPPFLAGS:=-Wall -Wextra -std=c++14 -fopenmp

# Directory containing the canopy library
CANOPY_DIR:=../include

all: canopy_example

canopy_example: canopy_example.cpp
	$(CPP) $(CPPFLAGS) -I$(CANOPY_DIR) $^ -o $@

clean:
	rm canopy_example example_model.tr
