echo Clearing build and install directories
if [ -d build ]; then
  rm -r build
fi

if [ -d install ]; then
  rm -r install
fi

echo Creating build directory
mkdir build && cd build

# First argument specifies the generator for CMake
# If no generator specified, use default
echo Building library

# Check if the generator was passed (is the first argument non-empty)
if [ -z "$1" ]; then
  echo No generator passed - assuming CMake default
  cmake -DCMAKE_BUILD_TYPE=Release ..
else
  echo Using passed generator "$1"
  cmake -G "$1" -DCMAKE_BUILD_TYPE=Release ..
fi

echo Installing library
cmake --build . --target install --config Release

echo Cleaning build directory
cd .. && rm -r build

echo Done
