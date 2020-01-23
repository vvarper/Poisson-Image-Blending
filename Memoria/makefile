.PHONY: clean
CC=g++
LINK=-lm
SRC=src
BIN=bin

all: memoria.pdf

memoria.pdf: proyecto_poisson_blending.md
	pandoc proyecto_poisson_blending.md -o memoria.pdf --template eisvogel --highlight-style tango

clean:
	@echo Limpiando directorio $(BIN)/
	rm -f $(BIN)/*
