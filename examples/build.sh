# Script to build an example.
# The example folder is passed as the only non-positional argument of the script.
# Usage: ./build.sh [options] example
# Eg: ./build.sh -c Release mnist

build_configuration=Debug
# Check build type
while getopts ":c:" opt; do
  case $opt in
    c)
      if [[ $OPTARG = "Release" ]] || [[ $OPTARG = "Debug" ]]; then
        build_configuration=$OPTARG
      else
        echo ">>> Not a valid build configuration: $OPTARG"
        exit 1
      fi
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

shift $((OPTIND - 1))

# Check if an example was provided, if not, exit.
if [ -z "$1" ]; then
  echo ">>> No example was specified. Please pass it as an argument."
  exit 1
fi

# Try to cd into the project, if that fails, exit the script.
cd "$1"
if [[ $? -ne 0 ]]; then
    echo ">>> $1 is not a folder. Please provide a valid example."
    exit 1
fi

# Check if the build directory already exists.
if [ ! -d build ]; then
  # If it doesn't, create it and cd into it.
  echo ">>> Creating the build directory"
  mkdir build && cd build
else
  # If it does, clear it and cd into it.
  echo ">>> Cleaning the build directory"
  rm -r build/* && cd build
fi

echo ">>> Building "$1" in $build_configuration mode"
cmake -DCMAKE_BUILD_TYPE=$build_configuration ..
cmake --build . --config $build_configuration
