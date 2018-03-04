
GCC=g++

all: DEBUG= -DNDEBUG
all: mtg


alld: mtg

BUILD_DIR=../bin



clean:
	rm -f *.o


debug: DEBUG = -g -DDEBUG
debug: alld

OBJ_FILES_G=utils.o  tree.o  treebank.o  str_utils.o  grammar.o hash_utils.o  layers.o  logger.o  random_utils.o

FLAGS_GCC=-std=c++17 -O3 -Wall -Wno-sign-compare -Wno-deprecated $(DEBUG) $(WSTR) -I../lib

mtg: $(OBJ_FILES_G) mtg.cpp
	mkdir -p $(BUILD_DIR)
	echo done

%.o: %.cpp %.h
	$(GCC)       $(FLAGS_GCC)    -o $@ -c $<
