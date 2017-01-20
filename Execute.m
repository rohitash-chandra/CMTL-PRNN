 % by Rohitash Chandra, 2016: c.rohitash(at)gmail.com 
 %for multi-step time series predition  

% built on: https://github.com/rohitash-chandra/Cooperative-Coevolution-CMAES


clear all
clc

out1 = fopen('out1-10stepsFinal.txt', 'w');
out2 = fopen('out2-10stepsFinal.txt', 'w');

% Declare NN here

            

MaxFE = 200000  ; % 4bit  
 

 MinError = [0.001]; %Min Error for each problem
 NumProb = 1; % just test one problem (sunspot)
  ProbMin = [-5]; % initial pop range
  ProbMax = [5];


 NumSteps = 10; % max prediction horizon

  

MaxRun = 3; % number of runs (experiments)

for app = 1:NumProb 

for depth = 5:5:5
        
decom = 5;
        Suc = 0; % when minimum error is satisfied
        
        SucT1 = 0;
        SucT2 = 0;
        SucT3 = 0;
        
  for Run=1:MaxRun
           
          Input = 5;  %  
           
           Output = 1;  

           for t=1:NumSteps

              
            H(t) = ((t*4)/2) + 7;
            Topology{t} = [Input, H(t) , Output]; 
              
           end 
            
            H
           [Dimen, D] = SetCCNN(Topology{1},   H);
            
          
           
         Problem = 1;

         for step=1:NumSteps
                              %set data and NN
            [TrainInput{step}, TrainTarget{step}, ValidInput{step}, ValidTarget{step}, TestInput{step}, TestTarget{step}] = Data(app, step);

             net{step} = FNNetwork(  TrainInput{step}, TrainTarget{step}, ValidInput{step}, ValidTarget{step} ,  Topology{step}, decom); 
         end 
            
           
           PopSize = round((4+floor(3*log(D(end))))) % use D for larged Hidden neurons

                  depth = decom;
     
          CCGA = CooperativeCoevolution(PopSize,Dimen, decom, ProbMax, ProbMin);   
       
   
      
    LocalFE = 1000;
        
    phase =1 
    
    CurrentFE = 1;
    
    
    BestEr = ones(1,NumSteps) 
        
    while CurrentFE < MaxFE & phase < 300
        
        
     CCGA = CooperativeCoevolution.CCEvolution( CCGA, LocalFE * phase, depth,  MinError,  Topology,  net); % pass FNN as net
       
        TotalFE   =   CooperativeCoevolution.GetFE(CCGA)    ;
        SolOne =  CooperativeCoevolution.GetSolution(CCGA); % gives whole solution
        FitList =  CooperativeCoevolution.GetFitList(CCGA) ; 
    
     MinFit  = min(FitList);
      
     
      for step=1:NumSteps  
         T1Solution{step} = CooperativeCoevolution.SeprateTaskSolution(CCGA,step); 
      end
     
     for step=1:NumSteps 
       TotalEr(step) =   FitList(step); 
   
       if TotalEr(step) < BestEr(step)
          BestEr(step) = TotalEr(step) 
          BestSolution{step} = T1Solution{step}; 
       end
     end
     
      BestEr
      FitList
      
      
     CurrentFE = TotalFE  
     
     phase = phase +1 
     %trans
    end
     
    
  for s=1:NumSteps   
    net{s} = FNNetwork.SaveTrainedNet(net{s}, BestSolution{s},  Topology, s); 
    net{s} =  FNNetwork. TestRegressionNetwork(net{s}) ;  
    T{s}.Test(Run) = FNNetwork. GetTestRMSE(net{s}); 
     Fit{s}.Error(Run) = FNNetwork. GetTrainRMSE(net{s}); 
  end
       
 
      
      
     fprintf(out1,'%d %d  %d   %.6f %.6f %.6f %.6f %.6f  %.6f %.6f %.6f %.6f %.6f  \n',  app,  depth,   Run,        Fit{1}.Error(Run),    Fit{2}.Error(Run),    Fit{3}.Error(Run),    Fit{4}.Error(Run),   Fit{5}.Error(Run), Fit{6}.Error(Run),    Fit{7}.Error(Run),    Fit{8}.Error(Run),    Fit{9}.Error(Run),   Fit{10}.Error(Run));
    fprintf(out1,'%d    %d  %d  %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f    \n',  app,  depth,   Run,     T{1}.Test(Run),  T{2}.Test(Run),  T{3}.Test(Run), T{4}.Test(Run),  T{5}.Test(Run), T{6}.Test(Run),  T{7}.Test(Run),  T{8}.Test(Run), T{9}.Test(Run),  T{10}.Test(Run) );
    
  end 
  
  
  for st=1:NumSteps 
      
      
         MeanTrain(st) = mean(Fit{st}.Error) ;
         STDTrain(st) = 1.96 *(std(Fit{st}.Error)/sqrt(MaxRun)) ;
         MeanTest(st) = mean(T{st}.Test) ;
         STDTest(st) = 1.96 *(std(T{st}.Test)/sqrt(MaxRun)) ;
  end
%      

     testmean = sum(MeanTest)/NumSteps ;
     
     trainmean = sum(MeanTrain)/NumSteps ;
%          
     teststd =    sum(STDTest)/NumSteps  ;
     
 
     trainstd =    sum(STDTrain)/NumSteps  ;
     
     fprintf(out2, ' %d  %d  %.6f %.6f %.6f    %.6f   %.6f    %.6f   %.6f %.6f %.6f    %.6f   %.6f     \n',    app,  depth,      MeanTrain(1), MeanTrain(2),MeanTrain(3),MeanTrain(4),MeanTrain(5),MeanTrain(6), MeanTrain(7),MeanTrain(8),MeanTrain(9),MeanTrain(10), trainmean );
        fprintf(out2, '  %d  %d  %.6f %.6f %.6f   %.6f %.6f    %.6f   %.6f %.6f %.6f    %.6f   %.6f        \n',  app,  depth,     STDTrain(1), STDTrain(2),STDTrain(3),STDTrain(4),STDTrain(5) , STDTrain(6), STDTrain(7),STDTrain(8),STDTrain(9),STDTrain(10), trainstd );
     
     fprintf(out2, '   %d  %d %.6f %.6f %.6f    %.6f   %.6f    %.6f  %.6f %.6f %.6f    %.6f   %.6f       \n',    app,  depth,     MeanTest(1), MeanTest(2),MeanTest(3),MeanTest(4),MeanTest(5), MeanTest(6), MeanTest(7),MeanTest(8),MeanTest(9),MeanTest(10), testmean );
        fprintf(out2, '  %d  %d  %.6f %.6f %.6f   %.6f %.6f    %.6f    %.6f %.6f %.6f    %.6f   %.6f        \n',  app, depth,      STDTest(1), STDTest(2),STDTest(3),STDTest(4),STDTest(5) ,   STDTest(6), STDTest(7),STDTest(8),STDTest(9),STDTest(10),teststd );
end

end


fclose(out1);
fclose(out2);
