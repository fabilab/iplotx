#!/bin/sh

# Get the new version
github_release_name=${1}

# Strip v at the beginning
version=${github_release_name#v}

echo "Version to be set: ${version}"

# Actually substitute
sed -i "s/__version__ = \"[0-9\.]\+\"/__version__ = \"${version}\"/" iplotx/version.py

# Check
echo "Content of version.py:"
cat iplotx/version.py
