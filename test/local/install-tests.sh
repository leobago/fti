#!/bin/bash

dest='../../build/test/local'

cp testrunner $dest/
cp testengine $dest/
cp fti_cfg_mock.cfg $dest/

cp recovervar.* $dest/
cp dCP.* $dest/