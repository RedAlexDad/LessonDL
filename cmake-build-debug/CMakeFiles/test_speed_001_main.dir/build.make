# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /home/redalexdad/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /home/redalexdad/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/redalexdad/Документы/GitHub/LessonDL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/test_speed_001_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_speed_001_main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_speed_001_main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_speed_001_main.dir/flags.make

CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o: CMakeFiles/test_speed_001_main.dir/flags.make
CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o: /home/redalexdad/Документы/GitHub/LessonDL/test_speed_001/main.cpp
CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o: CMakeFiles/test_speed_001_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o -MF CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o.d -o CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o -c /home/redalexdad/Документы/GitHub/LessonDL/test_speed_001/main.cpp

CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/redalexdad/Документы/GitHub/LessonDL/test_speed_001/main.cpp > CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.i

CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/redalexdad/Документы/GitHub/LessonDL/test_speed_001/main.cpp -o CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.s

# Object files for target test_speed_001_main
test_speed_001_main_OBJECTS = \
"CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o"

# External object files for target test_speed_001_main
test_speed_001_main_EXTERNAL_OBJECTS =

test_speed_001_main: CMakeFiles/test_speed_001_main.dir/test_speed_001/main.cpp.o
test_speed_001_main: CMakeFiles/test_speed_001_main.dir/build.make
test_speed_001_main: /usr/lib/x86_64-linux-gnu/libOpenCL.so
test_speed_001_main: CMakeFiles/test_speed_001_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_speed_001_main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_speed_001_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_speed_001_main.dir/build: test_speed_001_main
.PHONY : CMakeFiles/test_speed_001_main.dir/build

CMakeFiles/test_speed_001_main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_speed_001_main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_speed_001_main.dir/clean

CMakeFiles/test_speed_001_main.dir/depend:
	cd /home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL /home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug /home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug /home/redalexdad/Документы/GitHub/LessonDL/cmake-build-debug/CMakeFiles/test_speed_001_main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_speed_001_main.dir/depend

