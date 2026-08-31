#!/bin/sh

if [ -z "$1" ] ; then
    echo "Minimum Job Number argument is required.  Run as '$0 jobnum'"
    exit 1
fi

minjobnum="$1"

myself="$(id -u -n)"

for j in $(bjobs -o "jobid" --user="$myself") ; do
  if [ "$j" -gt "$minjobnum" ] ; then
    bkill "$j"
  fi
done