build_configuration=Debug
generator=""
# Check build type and see if generator was passed
while getopts ":c:g:" opt; do
  case $opt in
    c)
      if [[ $OPTARG = "Release" ]] || [[ $OPTARG = "Debug" ]]; then
        build_configuration=$OPTARG
      else
        echo ">>> Not a valid build configuration: $OPTARG"
        exit 1
      fi
      ;;
    g)
      generator=$OPTARG
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

echo ">>> Clearing build and install directories"
if [ -d build ]; then
  rm -r build
fi

if [ -d install ]; then
  rm -r install
fi

echo ">>> Creating build directory"
mkdir build && cd build

echo ">>> Building nnlib in $build_configuration mode"

# Check if the generator was passed
if [[ $generator = "" ]]; then
  echo ">>> No generator passed - assuming CMake default"
  cmake -DCMAKE_BUILD_TYPE=$build_configuration ..
else
  echo ">>> Using passed generator $generator"
  cmake -G "$generator" -DCMAKE_BUILD_TYPE=$build_configuration ..
fi

echo ">>> Installing library"
cmake --build . --target install --config $build_configuration

echo ">>> Cleaning build directory"
cd .. && rm -r build

echo ">>> Done"
