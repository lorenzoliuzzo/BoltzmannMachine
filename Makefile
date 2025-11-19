CXX := g++
CXXFLAGS := -std=c++23 -O2 -Wall -Wextra
EIGEN_PATH := /usr/local/include/eigen-5.0.0

TARGET := main
SOURCES := main.cpp
HEADERS := boltzmann.hpp

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(EIGEN_PATH) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)