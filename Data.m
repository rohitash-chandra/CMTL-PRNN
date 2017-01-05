function [TrainInput, TrainTarget, ValidInput, ValidTarget, TestInput, TestTarget] = Data(problem, output)
  
 if problem ==1
 load train.txt;  
load  validation.txt;   
load test.txt;
 end
  
 
 
 TRAIN=train;
VALID=validation;
TEST= test; 
 
  
input = 5;
maxout = 10; 

% for out =1:maxout 
  data  =  TEST; %targettrain, targettest
%   datainput{out} = data(1:end, 1:end -maxout);
%   dataout{out} = data(1:end, input+out:input+out);
% end
 
    trainA = []; 
  trainA = [TRAIN];

   Tartrain = []; 
  Tartrain = [VALID];

  TrainInput =   trainA(1:end, 1:end -maxout); 
  TrainTarget =  trainA(1:end, input+output:input+output); 


  ValidInput =  Tartrain(1:end, 1:end -maxout); 
  ValidTarget = Tartrain(1:end, input+output:input+output); 
  
  TestInput =  data(1:end-5, 1:end -maxout); 
  TestTarget =  data(1:end-5, input+output:input+output);
  
% %   TrainInput 
% %   TrainTarget
% %   ValidInput
% %   ValidTarget
% %   TestInput
% %   TestTarget
end
