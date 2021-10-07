#!/bin/sh

set -e

cd `dirname $0`

./run.sh tests/test.sh $*
