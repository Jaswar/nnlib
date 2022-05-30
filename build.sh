generator=""
# See if a generator was passed
while getopts ":g:" opt; do
  case $opt in
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

echo ">>> Clearing the install directory"
if [ -d install ]; then
  rm -r install
fi

function build_and_install {
  if [ -d build ]; then
    echo ">>> Clearing the build directory"
    rm -r build
  fi

  echo ">>> Creating build directory"
  mkdir build && cd build

  echo ">>> Building nnlib in $1 mode"

  # Check if the generator was passed
  if [[ $2 = "" ]]; then
    echo ">>> No generator passed - assuming CMake default"
    cmake -DCMAKE_BUILD_TYPE=$1 ..
  else
    echo ">>> Using passed generator $2"
    cmake -G "$2" -DCMAKE_BUILD_TYPE=$1 ..
  fi

  echo ">>> Installing library"
  cmake --build . --target install --config $1

  echo ">>> Clearing the build directory"
  cd .. && rm -r build
}

build_and_install "Debug" "$generator"
build_and_install "Release" "$generator"

echo ">>> Done"
