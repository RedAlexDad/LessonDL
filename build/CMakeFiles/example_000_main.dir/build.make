# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/redalexdad/Документы/GitHub/LessonDL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/redalexdad/Документы/GitHub/LessonDL/build

# Include any dependencies generated for this target.
include CMakeFiles/example_000_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/example_000_main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/example_000_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example_000_main.dir/flags.make

CMakeFiles/example_000_main.dir/example_000/main.cpp.o: CMakeFiles/example_000_main.dir/flags.make
CMakeFiles/example_000_main.dir/example_000/main.cpp.o: /home/redalexdad/Документы/GitHub/LessonDL/example_000/main.cpp
CMakeFiles/example_000_main.dir/example_000/main.cpp.o: CMakeFiles/example_000_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example_000_main.dir/example_000/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example_000_main.dir/example_000/main.cpp.o -MF CMakeFiles/example_000_main.dir/example_000/main.cpp.o.d -o CMakeFiles/example_000_main.dir/example_000/main.cpp.o -c /home/redalexdad/Документы/GitHub/LessonDL/example_000/main.cpp

CMakeFiles/example_000_main.dir/example_000/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/example_000_main.dir/example_000/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/redalexdad/Документы/GitHub/LessonDL/example_000/main.cpp > CMakeFiles/example_000_main.dir/example_000/main.cpp.i

CMakeFiles/example_000_main.dir/example_000/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/example_000_main.dir/example_000/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/redalexdad/Документы/GitHub/LessonDL/example_000/main.cpp -o CMakeFiles/example_000_main.dir/example_000/main.cpp.s

# Object files for target example_000_main
example_000_main_OBJECTS = \
"CMakeFiles/example_000_main.dir/example_000/main.cpp.o"

# External object files for target example_000_main
example_000_main_EXTERNAL_OBJECTS =

example_000_main: CMakeFiles/example_000_main.dir/example_000/main.cpp.o
example_000_main: CMakeFiles/example_000_main.dir/build.make
example_000_main: /usr/lib/x86_64-linux-gnu/libOpenCL.so
example_000_main: CMakeFiles/example_000_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_000_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_000_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example_000_main.dir/build: example_000_main
.PHONY : CMakeFiles/example_000_main.dir/build

CMakeFiles/example_000_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example_000_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example_000_main.dir/clean

CMakeFiles/example_000_main.dir/depend:
	cd /home/redalexdad/Документы/GitHub/LessonDL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL/build /home/redalexdad/Документы/GitHub/LessonDL/build /home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles/example_000_main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/example_000_main.dir/depend

