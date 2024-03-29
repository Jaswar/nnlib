path_to_executable=""
# See if the path to clang-format executable was passed
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
  echo ">>> Using $path_to_executable as the clang-format executable"
fi

# Navigate to correct directory if script was not moved from "scripts"
parent_dir=$(pwd | xargs basename)
if [[ $parent_dir = "scripts" ]]; then
  cd ..
fi

# Format all files
regex="\(\./\(src\|include\)/.*\.\(cpp\|cu\|h\|cuh\)\)\|\(\./test/\(assertions\|utils\)\.\(cpp\|h\)\)"
num_files=$(find . -regex $regex | wc -l)
echo ">>> Detected $num_files files to run auto-fix on"

should_fail=false
files=$(find . -regex $regex)
for file in $files; do
    echo ">>> Running auto-fix on $file"
    if [[ $path_to_executable = "" ]]; then
      clang-format -i -style=file $file
    else
      $path_to_executable -i -style=file $file
    fi

    if [[ $? -ne 0 ]]; then
	      echo ">>> Formatting incorrect in $file"
        should_fail=true
    fi
done

if [[ "$should_fail" = "true" ]]; then
  echo ">>> Formatting was incorrect in one or more files"
  exit 1
fi

echo ">>> Done"
