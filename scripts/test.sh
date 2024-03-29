architecture=""
# See if the path to clang-tidy executable was passed
while getopts ":a:" opt; do
  case $opt in
    a)
      architecture=$OPTARG
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

# Navigate to correct directory if script was not moved from "scripts"
parent_dir=$(pwd | xargs basename)
if [[ $parent_dir = "scripts" ]]; then
  cd ..
fi

# Clean build directory if it exists
if [[ -d build ]]; then
  echo ">>> Clearing build directory"
  rm -rf build
fi

# Create the build directory and cd into it
echo ">>> Creating build directory"
mkdir build && cd build

# Build the library with testing enabled
echo ">>> Building the library for testing"
cmake -DCMAKE_TEST_NNLIB=ON -DCMAKE_FORCE_ARCHITECTURE=$architecture ..
cmake --build .

# Execute the tests using ctest
echo ">>> Running the tests"
cd test && ctest

if [[ $? -ne 0 ]]; then
  echo ">>> Tests failed"
  exit 1
fi

# Clear the build directory once everything is done
echo ">>> Removing the build directory"
cd ..
rm -rf build

echo ">>> Done"
