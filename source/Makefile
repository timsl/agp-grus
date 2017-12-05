GCC     := g++
CLANG   := clang++
CFLAGS   := -std=c++11 -Wall --pedantic
CPP     := $(CLANG) $(CFLAGS)

BASEDIR := $(PWD)
LIBDIR  := -L $(BASEDIR)/bin/glad/lib64
BINDIR  := $(BASEDIR)/bin
OBJDIR  := $(BINDIR)/obj
SRCDIRS := src/ src/util/ src/sphere/ src/state/ src/input/
INCDIR  := $(addprefix -I$(BASEDIR)/,$(SRCDIRS)) -I$(BASEDIR)/glad/include/glad/

LIBS    := -lglad -lglfw -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread
MKDIR   := mkdir -p

VPATH   := $(BINDIR) $(OBJDIR) $(SRCDIRS)
SRCS    := $(wildcard $(BASEDIR)/src/**/*.cpp)
OBJS    := $(patsubst %.cpp,%.o,$(notdir $(SRCS))) bin/glad/glad.o

all: setup main

bin/glad/glad.o:
	cd glad && $(MAKE) BUILDDIR=../bin/glad

main: $(OBJS) main.cpp
	$(CPP) $(INCDIR) $(LIBDIR) src/main.cpp -o $(BINDIR)/main $(OBJDIR)/*.o $(LIBS)

%.o: %.cpp %.hpp
	$(CPP) $(INCDIR) -c $< -o $(OBJDIR)/$@

# additional dependencies
main sphere.o state.o util.o input.o: common.hpp

setup:
	$(MKDIR) $(BINDIR) $(OBJDIR)

clean:
	$(RM) -rf $(BINDIR) $(OBJDIR)

rebuild: clean all

ec: 
	# $^
	# $(VPATH)
	# $(SRCS)
	# $(OBJS)
