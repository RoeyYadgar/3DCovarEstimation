#!/bin/bash

# File containing the list of download links
links_file=$1

# Directory to store downloaded files
download_dir=$2
mkdir -p "$download_dir"


# Read each link from the file and process it
while IFS= read -r link; do
    # Get the file name from the link
    file_name=$(basename "$link")

    # Download the file
    wget -O "$download_dir/$file_name" "$link"

    # Extract the file
    gzip -d "$download_dir/$file_name"
done < "$links_file"

echo "Download and extraction completed."
