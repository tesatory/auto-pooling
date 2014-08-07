#!/bin/sh
## Last modified: 2011/10/4

if [ `uname -p` = i686 ]
then
  export MATLABROOT=/usr/share/matlab
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MATLABROOT/bin/glnx86:$MATLABROOT/extern/lib/glnx86:$MATLABROOT/sys/os/glnx86
else
  export MATLABROOT=/usr/local/MATLAB/R2011a
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MATLABROOT/bin/glnxa64:$MATLABROOT/extern/lib/glnxa64:$MATLABROOT/sys/os/glnxa64:$MATLABROOT/runtime/glnxa64
fi

printf "\n# # # start MATLAB and compile the file\n"
matlab -r "mcc -m $1;exit"
printf "\n# # # finished compiling and MATLAB successfully\n\n# # # run the program automatically.\n\n"
./$@
