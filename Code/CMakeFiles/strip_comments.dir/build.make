# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake

# The command to remove a file.
RM = /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code

# Utility rule file for strip_comments.

# Include any custom commands dependencies for this target.
include CMakeFiles/strip_comments.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/strip_comments.dir/progress.make

CMakeFiles/strip_comments:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "strip comments"
	/usr/bin/perl -pi -e 's#^[ \t]*//.*\n##g;' Wave-1D.cc

strip_comments: CMakeFiles/strip_comments
strip_comments: CMakeFiles/strip_comments.dir/build.make
.PHONY : strip_comments

# Rule to build all files generated by this target.
CMakeFiles/strip_comments.dir/build: strip_comments
.PHONY : CMakeFiles/strip_comments.dir/build

CMakeFiles/strip_comments.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/strip_comments.dir/cmake_clean.cmake
.PHONY : CMakeFiles/strip_comments.dir/clean

CMakeFiles/strip_comments.dir/depend:
	cd /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code/CMakeFiles/strip_comments.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/strip_comments.dir/depend

