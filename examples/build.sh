# Script to build an example.
# The example folder is passed as the first argument of the script.

# Check if the example was provided, if not, exit.
if [ -z "$1" ]; then
  echo No example was specified. Please pass it as an argument.
  exit 1
fi

# Try to cd into the project, if that fails, exit the script.
cd "$1"
if [[ $? -ne 0 ]]; then
    echo Please provide a valid example folder.
    exit 1
fi

# Check if the build directory already exists.
if [ ! -d build ]; then
  # If it doesn't, create it and cd into it.
  echo Creating the build directory
  mkdir build && cd build
else
  # If it does, clear it and cd into it.
  echo Cleaning the build directory
  rm -r build/* && cd build
fi

echo Building "$1"
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
