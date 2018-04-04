
GCC=g++

all: DEBUG= -DNDEBUG
all: main

BUILD_DIR=../bin

clean:
	rm -f *.o


debug: DEBUG = -g -DDEBUG
debug: alld

OBJ_FILES=utils.o str_utils.o hash_utils.o  layers.o  logger.o  random_utils.o conll_utils.o neural_encoder.o neural_net_hyperparameters.o bilstm_tagger.o

FLAGS_GCC=-std=c++17 -O3 -Wall -Wno-sign-compare -Wno-deprecated $(DEBUG) $(WSTR) -I../lib

main: $(OBJ_FILES) main.cpp
	mkdir -p $(BUILD_DIR)
	$(GCC)       $(FLAGS_GCC)   $(OBJ_FILES)   main.cpp   -o $(BUILD_DIR)/main

%.o: %.cpp %.h
	$(GCC)       $(FLAGS_GCC)    -o $@ -c $<
