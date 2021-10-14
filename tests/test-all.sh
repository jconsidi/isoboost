#!/bin/sh

set -e

cd `dirname $0`/..

tests/test.sh $*
tests/black.sh
