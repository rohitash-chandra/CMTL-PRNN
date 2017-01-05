function [TrainInput, TrainTarget, ValidInput, ValidTarget, TestInput, TestTarget] = Data(problem, output)
  
 if problem ==1
 load Sunspot/train.txt;  
load  Sunspot/validation.txt;   
load Sunspot/test.txt;
 end
 
 if problem ==2
     
 load Lazer/train.txt;  
load  Lazer/validation.txt;   
load Lazer/test.txt;
      
 end
 
  if problem ==4
 load Lorenz/train.txt;  
load  Lorenz/validation.txt;   
load Lorenz/test.txt;
 end
 
 if problem ==3
     
 load Mackey/train.txt;  
load  Mackey/validation.txt;   
load  Mackey/test.txt;
      
 end
 
 if problem ==5
     
 load Henon/train.txt;  
load  Henon/validation.txt;   
load  Henon/test.txt;
      
 end
 
 if problem ==6
     
 load ACFinance/train.txt;  
load  ACFinance/validation.txt;   
load  ACFinance/test.txt;
      
 end
 
 if problem ==7
 

 load Rossler/train.txt;  
load  Rossler/validation.txt;   
load  Rossler/test.txt;
      
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
