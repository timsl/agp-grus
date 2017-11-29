
CPP      = g++ -std=c++11 -Wall --pedantic
AR       = ar
INCDIR   = ./include
BUILDDIR = ./build
BUILDLIB = $(BUILDDIR)/lib64
CFLAGS   = -I$(INCDIR)
CP       = cp -rf
MKDIR    = mkdir -p

all: setup libglad.a
	
libglad.a: glad.o
	@$(AR) -cq $(BUILDLIB)/libglad.a $(BUILDDIR)/*.o
	@$(CP) $(INCDIR) $(BUILDDIR)
	
glad.o: src/glad.c
	@$(CPP) $(CFLAGS) -c src/glad.c -o $(BUILDDIR)/glad.o
	
setup:
	@$(MKDIR) $(BUILDLIB)

clean:
	@$(RM) -rf $(BUILDDIR)
	
rebuild: clean all

