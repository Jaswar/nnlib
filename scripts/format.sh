regex="\(\./\(src\|include\)/.*\.\(cpp\|cu\|h\|cuh\)\)\|\(\./test/\(assertions\|utils\)\.\(cpp\|h\)\)"
num_files=$(find . -regex $regex | wc -l)
echo ">>> Detected $num_files files to lint"

files=$(find . -regex $regex)
echo ">>> Linting $(echo $files | wc -l) files"
for file in $files; do
    echo ">>> Linting $file"
    clang-format --dry-run $file
	
    if [[ $? -ne 0 ]]; then
	echo ">>> Linter failed for $file"
        exit $?
    fi
done

echo ">>> Done"
