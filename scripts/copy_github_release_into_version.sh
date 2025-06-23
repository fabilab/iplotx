#!/bin/sh

github_release_name=${1}
# strip v at the beginning
version=${github_release_name#v}

sed -i "s/__version__ = \"[0-9\.]\+\"/__version__ = \"${version}\"/" iplotx/version.py
