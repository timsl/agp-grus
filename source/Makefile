GCC     := g++
CLANG   := clang++
CFLAGS   := -std=c++11 -Wall --pedantic
CPP     := $(CLANG) $(CFLAGS)

BASEDIR := $(PWD)
LIBDIR  := -L $(BASEDIR)/bin/glad/lib64
BINDIR  := $(BASEDIR)/bin
OBJDIR  := $(BINDIR)/obj
SRCDIRSR:= src/ src/util/ src/sphere/ src/state/ src/input/
SRCDIRS := $(addprefix $(BASEDIR)/,$(SRCDIRSR))
INCDIR  := $(addprefix -I,$(SRCDIRS)) -I$(BASEDIR)/glad/include/glad/

LIBS    := -lglad -lglfw -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread
MKDIR   := mkdir -p

VPATH   := $(BINDIR) $(OBJDIR) $(SRCDIRS)
SRCS    := $(wildcard $(BASEDIR)/src/**/*.cpp $(BASEDIR)/src/*.cpp)
OBJS    := $(patsubst %.cpp,%.o,$(notdir $(SRCS))) bin/glad/glad.o

all: setup moon

bin/glad/glad.o:
	cd glad && $(MAKE) BUILDDIR=../bin/glad

moon: $(OBJS)
	$(CPP) $(INCDIR) $(LIBDIR) $(OBJDIR)/*.o -o $(BINDIR)/moon $(LIBS)

%.o: %.cpp
	$(CPP) $(INCDIR) -c $< -o $(OBJDIR)/$@

# additional dependencies
main.o sphere.o state.o util.o input.o: common.hpp bin/glad/glad.o
state.o input.o: heldactions.hpp
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
