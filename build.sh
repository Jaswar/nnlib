echo Clearing build and install directories
rm -r build
rm -r install

echo Creating build directory
mkdir build && cd build

# First argument specifies the generator for CMake
# If no generator specified, use default
echo Building library

# Check if the generator was passed (is the first argument non-empty)
if [ -z "$1" ]
then
  cmake ..
else
  cmake -G "$1" ..
fi

echo Installing library
cmake --build . --target install

echo Cleaning build directory
cd .. && rm -r build