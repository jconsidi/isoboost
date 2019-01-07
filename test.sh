#!/bin/sh

set -e

cd `dirname $0`

./run.sh test/test.sh $*
