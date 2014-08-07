%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sample code of Matlab compile (last update 2012/4/11)
% 
% 1. compile (in Matlab):
%  $mcc -m sample_func
%
% 2. execution example (in command terminal):
%  $./sample_func 5 save_data.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function sample_func (num, savefile)

  if(isdeployed)
    num = str2num(num);
  end

  x=1;
  for i = 1:num
    x=x*i;
  end

  display(x);
  save(savefile,'x');

return
