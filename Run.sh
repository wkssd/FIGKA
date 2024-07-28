#!/bin/bash

# 定义要运行的脚本
scripts=("CNNv8.1.py" "CNNv8.2.py" "CNNv8.3.py" "CNNv8.4.py")

# 依次运行每个脚本
for script in "${scripts[@]}"
do
  echo "Running $script..."
  python "$script"
  if [ $? -ne 0 ]; then
    echo "Error running $script"
    exit 1
  fi
done

echo "All scripts executed."


