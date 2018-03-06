#!/usr/bin/env bash
# prerequisites: imagemagick & gnome

difference=$(mktemp /tmp/difference.XXXXXXXXX.png)
montage=$(mktemp /tmp/montage.XXXXXXXXX.png)

convert "$1" "$2" -alpha off +repage \
\( -clone 0 -clone 1 -compose difference -composite -threshold 0 \) \
-delete 1 -alpha off -compose copy_opacity -composite -trim \
"$difference"

montage -label %f -pointsize 60 -geometry +2+2 -bordercolor silver "$1" "$2" "$difference" "$montage"

eog "$montage"

rm "$difference"
rm "$montage"

