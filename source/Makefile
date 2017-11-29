
CPP     = clang++ -std=c++11 -Wall --pedantic
INCDIR  = -I./ -I./util -I./sphere -I./modules/build/include
LIBDIR  = -L./ -L./modules/build/lib
BINDIR  = ./bin
OBJDIR  = $(BINDIR)/obj
LIBS    = -lglad -lglfw3 -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread
CFLAGS  = $(INCDIR) $(LIBDIR)
MKDIR   = mkdir -p

all: setup main.out

main.out: util.o sphere.o
	@$(CPP) $(CFLAGS) main.cpp -o $(BINDIR)/main.out $(OBJDIR)/*.o $(LIBS)

# The following line is an example on how to compile your own clases. Use
# the $(OBJDIR) directory to output your object files:
#
# example_class.o:
#	@$(CPP) $(CFLAGS) -c example_class.cpp \
#					  -o $(OBJDIR)/example_class.o
#
# Do not forget to specify example_class.o alongside the main.out label. This
# way, we tell make that main.out depends on example_class.o, triggering the
# build beforehand.

sphere.o:
	@$(CPP) $(CFLAGS) -c sphere/sphere.cpp -o $(OBJDIR)/sphere.o

util.o:
	@$(CPP) $(CFLAGS) -c util/util.cpp -o $(OBJDIR)/util.o

setup:
	@$(MKDIR) $(BINDIR) $(OBJDIR)

clean:
	@$(RM) -rf $(BINDIR) $(OBJDIR)

rebuild: clean all

