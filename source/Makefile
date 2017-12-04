CPP     = ${GCC}
GCC     = g++ -std=c++11 -Wall --pedantic
CLANG   = clang++ -std=c++11 -Wall --pedantic
INCDIR  = -I./src/ -I./src/util -I./src/sphere -I./src/state -I./glad/include/glad/
LIBDIR  = -L ./bin/glad/lib64
BINDIR  = ./bin
OBJDIR  = $(BINDIR)/obj
LIBS    = -lglad -lglfw -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread
MKDIR   = mkdir -p

all: setup bin/main.out

bin/glad/glad.o:
	cd glad && $(MAKE) BUILDDIR=../bin/glad

bin/main.out: $(OBJDIR)/util.o $(OBJDIR)/sphere.o $(OBJDIR)/state.o $(BINDIR)/glad/glad.o
	$(CPP) $(INCDIR) $(LIBDIR) src/main.cpp -o $(BINDIR)/main.out $(OBJDIR)/*.o $(LIBS)
	chmod a+x $(BINDIR)/main.out

$(OBJDIR)/state.o:
	$(CPP) $(INCDIR) -c src/state/state.cpp -o $@

$(OBJDIR)/sphere.o:
	$(CPP) $(INCDIR) -c src/sphere/sphere.cpp -o $@

$(OBJDIR)/util.o:
	$(CPP) $(INCDIR) -c src/util/util.cpp -o $@

setup:
	$(MKDIR) $(BINDIR) $(OBJDIR)

clean:
	$(RM) -rf $(BINDIR) $(OBJDIR)

rebuild: clean all

