# Script to build an example.
# The example folder is passed as the only non-positional argument of the script.
# Usage: ./build.sh [options] example
# Eg: ./build.sh -c Release mnist

build_configuration=Debug
prefix_path=""
# Check build type
while getopts ":c:p:" opt; do
  case $opt in
    c)
      build_configuration=$OPTARG
      ;;
    p)
      prefix_path=$OPTARG
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

# Check if the prefix path is specified
if [[ $prefix_path = "" ]]; then
  echo ">>> Please specify prefix path using -p."
  exit 1
fi

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
cmake -DCMAKE_BUILD_TYPE=$build_configuration -DCMAKE_PREFIX_PATH="$prefix_path" ..
cmake --build . --config $build_configuration
