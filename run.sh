#!/bin/sh

set -e

cd `dirname $0`

IMAGE=`basename $PWD`

docker build -t "$IMAGE" .
docker run -t -i --rm "$IMAGE" $*
