#!/bin/bash

# original file from 'https://gist.github.com/pete-otaqui/4188238'

# modified by Kai Keller to adjust for FTI repository

# works with a file called VERSION in the current directory,
# the contents of which should be a semantic version number
# such as "1.2.3"

# this script will display the current version, automatically
# suggest a "minor" version update, and ask for input to use
# the suggestion, or a newly entered value.

# once the new version number is determined, the script will
# pull a list of changes from git history, prepend this to
# a file called CHANGELOG (under the title of the new version
# number) and create a GIT tag.

if [ -f VERSION ]; then
    BASE_STRING=`cat VERSION`
    BASE_LIST=(`echo $BASE_STRING | tr '.' ' '`)
    V_MAJOR=${BASE_LIST[0]}
    V_MINOR=${BASE_LIST[1]}
    V_PATCH=${BASE_LIST[2]}
    echo "Current version : $BASE_STRING"
    V_PATCH=$((V_PATCH + 1))
    SUGGESTED_VERSION="$V_MAJOR.$V_MINOR.$V_PATCH"
    read -p "Enter a version number [$SUGGESTED_VERSION]: " INPUT_STRING
    if [ "$INPUT_STRING" = "" ]; then
        INPUT_STRING=$SUGGESTED_VERSION
    fi
    echo -e "\nChanging Version number to $INPUT_STRING.\n\n[WARNING]\nThis includes changes to CMakeList.txt\nadding a new tag to the repository\nand push the new tag to the repository.\n"
	read -p "Change version number now [y/n]?" -n 1 -r
	echo    # (optional) move to a new line
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
    	echo "Setting version to $INPUT_STRING"
    	echo $INPUT_STRING > VERSION
		sed -i 's@^\(project.*VERSION \)\([0-9]\.[0-9]\.[0-9]\)\( LANGUAGES.*\)$@\1'"`cat VERSION`"'\3@g' CMakeLists.txt
		sed -n 's@^\(project.*VERSION \)\([0-9]\.[0-9]\.[0-9]\)\( LANGUAGES.*\)$@\1\2\3@p' CMakeLists.txt
    	echo "Version $INPUT_STRING:" > tmpfile
    	git log --pretty=format:" - %s" "v$BASE_STRING"...HEAD >> tmpfile
    	echo "" >> tmpfile
    	echo "" >> tmpfile
    	cat CHANGELOG >> tmpfile
    	mv tmpfile CHANGELOG
        set -x
        cat CHANGELOG
        cat VERSION
    	git add CHANGELOG VERSION CMakeLists.txt
    	git commit -m "Version bump to $INPUT_STRING"
    	git tag -a -m "Tagging version $INPUT_STRING" "v$INPUT_STRING"
    	git push origin --tags
        set +x
	elif [[ ! $REPLY =~ ^[Nn]$ ]]
	then
		echo "answer either with 'y' or 'n'! (nothing happened yet...)"
	fi
else
    echo "Could not find a VERSION file!"
fi
