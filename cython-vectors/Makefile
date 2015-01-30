PYTHON := python-dbg
PYVERSION := $(shell $(PYTHON) -c "import sys; print(sys.version[:3])")

INCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
PLATINCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc(plat_specific=True))")
LIBDIR1 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LIBDIR2 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBPL'))")
PYLIB := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBRARY')[3:-2])")

LINKCC := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKCC'))")
LINKFORSHARED := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKFORSHARED'))")
LIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LIBS'))")
SYSLIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SYSLIBS'))")

CFLAGS := -w -O1 -fPIC -fopenmp -std=c++11 -g
LINKFLAGS := -lgsl -lgslcblas -lm -fopenmp -g

# CFLAGS += -isystem/usr/include/x86_64-linux-gnu/c++/4.8



default: cyvec.so

.SECONDARY:
%.so: %.o
	g++ -o $@ $^ -L$(LIBDIR1) -L$(LIBDIR2) -l$(PYLIB) $(LIBS) $(SYSLIBS) $(LINKFORSHARED) $(LINKFLAGS) -shared
%.cpp: %.pyx
	cython --cplus --gdb $^
%.o: %.cpp
	g++ $(CFLAGS) -c -o $@ $^ -I$(INCDIR) -I$(PLATINCDIR)
clean:
	rm -vf *.o *.c *.cpp cyvec.so
