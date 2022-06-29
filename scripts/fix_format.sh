# Navigate to correct directory if script was not moved from "scripts"
parent_dir=$(pwd | xargs basename)
if [[ $parent_dir = "scripts" ]]; then
  cd ..
fi

# Format all files
regex="\(\./\(src\|include\)/.*\.\(cpp\|cu\|h\|cuh\)\)\|\(\./test/\(assertions\|utils\)\.\(cpp\|h\)\)"
num_files=$(find . -regex $regex | wc -l)
echo ">>> Detected $num_files files to run auto-fix on"

files=$(find . -regex $regex)
for file in $files; do
    echo ">>> Running auto-fix on $file"
    clang-format -i -style=file $file
done

echo ">>> Done"
