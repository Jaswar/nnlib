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

files=$(find . -regex $regex)
echo ">>> Linting $(echo $files | wc -l) files"
for file in $files; do
    echo ">>> Linting $file"
    clang-tidy --quiet -p build $file
	
    if [[ $? -ne 0 ]]; then
	echo ">>> Linter failed for $file"
        exit $?
    fi
done

# Clear the build directory once everything is done
echo ">>> Removing the build directory"
rm -rf build

echo ">>> Done"
