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

# Utility rule file for sign.

# Include any custom commands dependencies for this target.
include CMakeFiles/sign.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sign.dir/progress.make

CMakeFiles/sign:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Digitally signing Wave-1D"
	/Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo '' && /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo '***************************************************************************' && /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo '** Error: No Mac OSX developer certificate specified **' && /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo '** Please reconfigure with -DOSX_CERTIFICATE_NAME="<...>" **' && /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo '***************************************************************************' && /Applications/deal.II.app/Contents/Resources/spack/opt/spack/cmake-3.20.3-oa76/bin/cmake -E echo ''

sign: CMakeFiles/sign
sign: CMakeFiles/sign.dir/build.make
.PHONY : sign

# Rule to build all files generated by this target.
CMakeFiles/sign.dir/build: sign
.PHONY : CMakeFiles/sign.dir/build

CMakeFiles/sign.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sign.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sign.dir/clean

CMakeFiles/sign.dir/depend:
	cd /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code /Users/sergio/Documents/Classes/Finite_Element_Methods/Codes/Project/Code/CMakeFiles/sign.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sign.dir/depend
