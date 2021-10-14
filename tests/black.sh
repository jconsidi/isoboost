#!/bin/sh

set -e

cd `dirname $0`/..

black --check */*.py
