# Compiler settings
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -g -fopenmp

# The name of your final executable
EXEC = main

# The object files needed to build the executable
OBJS = main.o PRNG.o


all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)


main.o: main.cc PRNG.h
	$(CXX) $(CXXFLAGS) -c main.cc

PRNG.o: PRNG.cc PRNG.h
	$(CXX) $(CXXFLAGS) -c PRNG.cc


clean:
	rm -f $(OBJS) $(EXEC)

.PHONY: all clean