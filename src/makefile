
GCC=g++

all: DEBUG= -DNDEBUG

BUILD_DIR=../bin

clean:
	rm -f *.o


debug: DEBUG = -g -DDEBUG
debug: alld

OBJ_FILES=utils.o str_utils.o hash_utils.o  layers.o  logger.o  random_utils.o

FLAGS_GCC=-std=c++17 -O3 -Wall -Wno-sign-compare -Wno-deprecated $(DEBUG) $(WSTR) -I../lib

all: $(OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	echo done

%.o: %.cpp %.h
	$(GCC)       $(FLAGS_GCC)    -o $@ -c $<
