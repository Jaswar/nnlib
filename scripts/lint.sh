path_to_executable=""
# See if the path to clang-tidy executable was passed
while getopts ":p:" opt; do
  case $opt in
    p)
      path_to_executable=$OPTARG
      ;;
    \?)
      echo ">>> Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo ">>> Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ $path_to_executable != "" ]]; then
  echo ">>> Using $path_to_executable as the clang-tidy executable"
fi

# Navigate to correct directory if script was not moved from "scripts"
parent_dir=$(pwd | xargs basename)
if [[ $parent_dir = "scripts" ]]; then
  cd ..
fi

# Clean build directory if it exists
if [[ -d build ]]; then
    echo ">>> Clearning build directory"
    rm -rf build
fi

# Create the build directory and cd into it
echo ">>> Creating build directory"
mkdir build && cd build

# Build the library while exporting compilation database and testing
echo ">>> Building the library for linting"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_TEST_NNLIB=ON ..
cmake --build .

# Run clang tidy on every wanted file
echo ">>> Starting clang-tidy, warnings should be ignored"
cd ..

regex="\(\./\(src\|include\)/.*\.\(cpp\|cu\|h\|cuh\)\)\|\(\./test/\(assertions\|utils\)\.\(cpp\|h\)\)"
num_files=$(find . -regex $regex | wc -l)
echo ">>> Detected $num_files files to lint"

should_fail=false
files=$(find . -regex $regex)
for file in $files; do
    echo ">>> Linting $file"
    if [[ $path_to_executable = "" ]]; then
      clang-tidy --quiet -p build $file
    else
      $path_to_executable --quiet -p build $file
    fi
	
    if [[ $? -ne 0 ]]; then
	      echo ">>> Linter failed for $file"
        should_fail=true
    fi
done

if [[ "$should_fail" = "true" ]]; then
  echo ">>> Linter failed for one or more files"
  exit 1
fi

# Clear the build directory once everything is done
echo ">>> Removing the build directory"
rm -rf build

echo ">>> Done"
