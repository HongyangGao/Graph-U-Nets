dir_guard = @mkdir -p $(@D)
FIND := find
CXX := g++

# In this way, we can overide these two vars using export
# so different machines can compile.
ifeq ($(RD_LIB),)
	RD_LIB=$(RDBASE)/lib
endif
ifeq ($(RD_INC),)
	RD_INC=$(RDBASE)/Code
endif

CXXFLAGS += -Wall -O3 -std=c++11
LDFLAGS += -lm -L$(RD_LIB) -lRDKitGraphMol -lRDKitRDGeneral -lRDKitSmilesParse -lRDKitDepictor -lRDKitSmilesParse -lRDKitFileParsers -lRDKitFingerprints

include_dirs = ./include $(RD_INC)

CXXFLAGS += $(addprefix -I,$(include_dirs)) -Wno-unused-local-typedef
CXXFLAGS += -fPIC
cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -print | rev | cut -d"/" -f1 | rev)
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))

objs = $(addprefix build/lib/,$(cxx_obj_files))
DEPS = $(objs:.o=.d)

target = build/dll/libmol.so
target_dep = $(addsuffix .d,$(target))

.PRECIOUS: build/lib/%.o

all: $(target)

build/dll/libmol.so : src/mol_lib.cpp $(objs)
	$(dir_guard)
	$(CXX) -shared $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) $(LDFLAGS)

DEPS += $(target_dep)

build/lib/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^)

clean:
	rm -rf build

-include $(DEPS)
