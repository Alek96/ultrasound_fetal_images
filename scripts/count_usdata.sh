#!/bin/bash

US_VIDEOS_FOLDER=${1:-US_VIDEOS}

echo "Videos: $(find data/"${US_VIDEOS_FOLDER}"/videos/* -maxdepth 0 -type f  2>/dev/null | wc -l)"

echo "Selected videos: $(find data/"${US_VIDEOS_FOLDER}"/selected/* -maxdepth 0 -type d  2>/dev/null | wc -l)"
echo "Selected frames: $(find data/"${US_VIDEOS_FOLDER}"/selected/* -maxdepth 1 -type f  2>/dev/null | wc -l)"

echo "Labeled videos: $(find data/"${US_VIDEOS_FOLDER}"/labeled/* -maxdepth 0 -type d  2>/dev/null | wc -l)"
echo "Labeled frames: $(find data/"${US_VIDEOS_FOLDER}"/labeled/* -maxdepth 2 -type f  2>/dev/null | wc -l)"
