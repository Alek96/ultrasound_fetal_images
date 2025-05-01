#!/bin/bash

version_0="4"
version_1="10"
version_2="0"
version_3="82"
version="${version_0}.${version_1}.${version_2}"

# ---

version_current=$(pip show opencv-python | sed -n 's/.*Version: //p')
location=$(pip show opencv-python | sed -n 's/.*Location: //p')

if [[ "${version_current}" != "${version}" ]]; then
  echo "Cannot rename opencv-python version"
else
  echo "Rename opencv-python version"

  opencv_python_path="${location}/opencv_python-${version}.dist-info"
  opencv_python_new_path="${location}/opencv_python-${version}.${version_3}.dist-info"

  sed -i.bak "/^Version: ${version_0}\.${version_1}\.${version_2}/c\Version: ${version_0}\.${version_1}\.${version_2}\.${version_3}" "${opencv_python_path}/METADATA"
  mv "${opencv_python_path}" "${opencv_python_new_path}"
fi

# ---

version_current=$(pip show opencv-python-headless | sed -n 's/.*Version: //p')
location=$(pip show opencv-python-headless | sed -n 's/.*Location: //p')

if [[ "${version_current}" != "${version}" ]]; then
  echo "Cannot rename opencv-python-headless version"
else
  echo "Rename opencv-python-headless version"

  opencv_python_headless_path="${location}/opencv_python_headless-${version}.dist-info"
  opencv_python_headless_new_path="${location}/opencv_python_headless-${version}.${version_3}.dist-info"

  sed -i.bak "/^Version: ${version_0}\.${version_1}\.${version_2}/c\Version: ${version_0}\.${version_1}\.${version_2}\.${version_3}" "${opencv_python_headless_path}/METADATA"
  mv "${opencv_python_headless_path}" "${opencv_python_headless_new_path}"
fi
