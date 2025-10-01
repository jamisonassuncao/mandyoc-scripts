#!/usr/bin/env bash

# ---
# This script organizes simulation output files (.txt) into subdirectories.
# It creates a directory for each data type and moves the corresponding files into it.
# ---

# Define a function to handle the file moving logic
move_files() {
    # The pattern to match, e.g., "density*.txt"
    local pattern="$1"
    # The destination directory, e.g., "density"
    local dest_dir="$2"

    # Check if any files matching the pattern actually exist.
    # 'compgen -G' is a safe way to check for file patterns.
    if ! compgen -G "$pattern" > /dev/null; then
        echo "No files found for '$pattern'. Skipping."
        return
    fi

    # Create the destination directory. The '-p' flag prevents errors if it already exists.
    mkdir -p "$dest_dir"

    # Move the files and report success.
    mv $pattern "$dest_dir/"
    echo "Successful: Moved '$pattern' files to '$dest_dir/'"
}

# Use an array for the list of main variables. This is cleaner than a simple string.
prefixes=(
    "density" "heat" "pressure" "surface" "strain_rate"
    "strain" "temperature" "time" "viscosity" "velocity"
)

echo "Starting outputs organization"

# Loop through each prefix and call our function
for prefix in "${prefixes[@]}"; do
    move_files "${prefix}*.txt" "$prefix"
done

# Handle the special cases using the same reusable function
move_files "litho*.txt" "lithos"
move_files "step*.txt" "steps"

echo "Completed"
