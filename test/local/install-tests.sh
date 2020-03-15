#!/bin/bash

origin='/home/alexandre/Documents/git/contrib/fti/test/local'
dest='/home/alexandre/Documents/git/contrib/fti/build/test/local'

cp $origin/testrunner $dest/
cp $origin/testengine $dest/
cp $origin/fti_cfg_mock.cfg $dest/

cp $origin/recovervar.* $dest/