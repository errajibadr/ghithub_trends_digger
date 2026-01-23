#!/bin/bash
# Extract files from a concatenated output (reverse of concat_staged.sh)
# Usage: ./scripts/extract_concat.sh [input_file] [--dry-run]
#        Default input: staged_files.txt
#        --dry-run: Show what would be created without writing files

INPUT_FILE="${1:-staged_files.txt}"
DRY_RUN=false

# Check for --dry-run flag
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    fi
done

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Extracting files from: $INPUT_FILE"
if [ "$DRY_RUN" == true ]; then
    echo "[DRY RUN MODE - no files will be created]"
fi
echo ""

# State machine variables
current_file=""
in_content=false
content=""
files_created=0

while IFS= read -r line || [ -n "$line" ]; do
    if [ "$in_content" == false ]; then
        # Looking for filename (non-empty line that's not a separator)
        if [ -n "$line" ] && [ "$line" != "----" ] && [ "$line" != "-------" ]; then
            current_file="$line"
        elif [ "$line" == "----" ] && [ -n "$current_file" ]; then
            # Start of content
            in_content=true
            content=""
        fi
    else
        # In content mode
        if [ "$line" == "-------" ]; then
            # End of content - write file
            if [ -n "$current_file" ]; then
                # Remove trailing newline from content
                content="${content%$'\n'}"
                
                if [ "$DRY_RUN" == true ]; then
                    echo "[WOULD CREATE] $current_file ($(echo -n "$content" | wc -c | tr -d ' ') bytes)"
                else
                    # Create directory if needed
                    dir=$(dirname "$current_file")
                    if [ "$dir" != "." ] && [ ! -d "$dir" ]; then
                        mkdir -p "$dir"
                        echo "[MKDIR] $dir"
                    fi
                    
                    # Write content to file
                    printf '%s' "$content" > "$current_file"
                    echo "[CREATED] $current_file"
                fi
                ((files_created++))
            fi
            
            # Reset state
            current_file=""
            in_content=false
            content=""
        else
            # Append line to content
            if [ -n "$content" ]; then
                content="$content"$'\n'"$line"
            else
                content="$line"
            fi
        fi
    fi
done < "$INPUT_FILE"

echo ""
if [ "$DRY_RUN" == true ]; then
    echo "Would create $files_created file(s)"
else
    echo "Extracted $files_created file(s)"
fi
