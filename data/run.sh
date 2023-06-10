#!/bin/bash
i=1
for file in *.{jpg,png,gif} # Remplacez la liste des extensions de fichier par les types d'images que vous souhaitez renommer
do
    mv "$file" "$i.${file##*.}"
    ((i++))
done