# https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/
# -x cu and -dc are relevant stuff, but compiling .cpp as -x cu means we have to skip -pedantic due to
# https://stackoverflow.com/questions/31000996/style-of-line-directive-is-a-gcc-extension


NVCC    := nvcc -lineinfo -arch=sm_52 -use_fast_math -std=c++11 -O2
NCFLAGS := --compiler-options "-Wall -pedantic"
CLANG   := clang++ -std=c++11-Wall -pedantic
CPPF    := $(NVCC)

CUDASAMPLESLOC := /opt/cuda/samples/common/inc

BASEDIR := $(PWD)
LIBDIR  := -L $(BASEDIR)/bin/glad/lib64
BINDIR  := $(BASEDIR)/bin
OBJDIR  := $(BINDIR)/obj
SRCDIRSR:= src/ src/util/ src/sphere/ src/state/ src/input/ src/kernel/ src/gpustate/
SRCDIRS := $(addprefix $(BASEDIR)/,$(SRCDIRSR))
INCDIR  := $(addprefix -I,$(SRCDIRS)) -I$(BASEDIR)/glad/include/glad/ -I$(CUDASAMPLESLOC)

LIBS    := -lglad -lglfw -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -lpthread
MKDIR   := mkdir -p

VPATH   := $(BINDIR) $(OBJDIR) $(SRCDIRS)
SRCS    := $(wildcard $(BASEDIR)/src/**/*.cpp $(BASEDIR)/src/*.cpp)
OBJS    := $(patsubst %.cpp,%.o,$(notdir $(SRCS))) bin/glad/glad.o kernel.o gpustate.o

all: setup moon

bin/glad/glad.o:
	cd glad && $(MAKE) BUILDDIR=../bin/glad

moon: $(OBJS)
	$(CPPF) $(INCDIR) $(LIBDIR) $(OBJDIR)/*.o -o $(BINDIR)/moon $(LIBS)

%.o: %.cpp
	$(CPPF) $(INCDIR) $(NCFLAGS) -c $< -o $(OBJDIR)/$@

%.o: %.cu
	$(CPPF) $(INCDIR) -dc $< -o $(OBJDIR)/$@

# additional dependencies
main.o sphere.o state.o util.o input.o: common.hpp bin/glad/glad.o
state.o input.o: heldactions.hpp
main.o: input.hpp input.cpp sphere.cpp sphere.hpp input.cpp input.hpp util.hpp util.cpp state.cpp state.hpp
main.o input.o: kernel.cu kernel.cuh
kernel.o: gpustate.cuh gpustate.cu state.hpp state.cpp
%.o: %.hpp

setup:
	@$(MKDIR) $(BINDIR) $(OBJDIR)

clean:
	$(RM) -rf $(BINDIR) $(OBJDIR)

run: all
	$(BASEDIR)/bin/moon

rebuild: clean all

ec: 
	@echo $(INCDIR)
	@echo $(OBJS)
