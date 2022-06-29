# Navigate to correct directory if script was not moved from "scripts"
parent_dir=$(pwd | xargs basename)
if [[ $parent_dir = "scripts" ]]; then
  cd ..
fi

# Format all files
regex="\(\./\(src\|include\)/.*\.\(cpp\|cu\|h\|cuh\)\)\|\(\./test/\(assertions\|utils\)\.\(cpp\|h\)\)"
num_files=$(find . -regex $regex | wc -l)
echo ">>> Detected $num_files files to format"

files=$(find . -regex $regex)
echo ">>> Linting $(echo $files | wc -l) files"
for file in $files; do
    echo ">>> Linting $file"
    clang-format -i -style=file $file
	
    if [[ $? -ne 0 ]]; then
	echo ">>> Linter failed for $file"
        exit $?
    fi
done

echo ">>> Done"
