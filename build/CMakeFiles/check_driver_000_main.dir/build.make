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
include CMakeFiles/check_driver_000_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/check_driver_000_main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/check_driver_000_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/check_driver_000_main.dir/flags.make

CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o: CMakeFiles/check_driver_000_main.dir/flags.make
CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o: /home/redalexdad/Документы/GitHub/LessonDL/check_driver_000/main.cpp
CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o: CMakeFiles/check_driver_000_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o -MF CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o.d -o CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o -c /home/redalexdad/Документы/GitHub/LessonDL/check_driver_000/main.cpp

CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/redalexdad/Документы/GitHub/LessonDL/check_driver_000/main.cpp > CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.i

CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/redalexdad/Документы/GitHub/LessonDL/check_driver_000/main.cpp -o CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.s

# Object files for target check_driver_000_main
check_driver_000_main_OBJECTS = \
"CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o"

# External object files for target check_driver_000_main
check_driver_000_main_EXTERNAL_OBJECTS =

check_driver_000_main: CMakeFiles/check_driver_000_main.dir/check_driver_000/main.cpp.o
check_driver_000_main: CMakeFiles/check_driver_000_main.dir/build.make
check_driver_000_main: /usr/lib/x86_64-linux-gnu/libOpenCL.so
check_driver_000_main: CMakeFiles/check_driver_000_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable check_driver_000_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check_driver_000_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/check_driver_000_main.dir/build: check_driver_000_main
.PHONY : CMakeFiles/check_driver_000_main.dir/build

CMakeFiles/check_driver_000_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/check_driver_000_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/check_driver_000_main.dir/clean

CMakeFiles/check_driver_000_main.dir/depend:
	cd /home/redalexdad/Документы/GitHub/LessonDL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL/build /home/redalexdad/Документы/GitHub/LessonDL/build /home/redalexdad/Документы/GitHub/LessonDL/build/CMakeFiles/check_driver_000_main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/check_driver_000_main.dir/depend

