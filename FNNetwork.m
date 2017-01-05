
%---------------------------------------------------------
% FNNetwork in OOP by Dr. Rohitash Chandra, 2016: c.rohitash@gmail.com
%

%to use later: http://www.codeproject.com/Articles/55691/Neural-Network-Learning-by-the-Levenberg-Marquardt.aspx
%--------------------------------------------------------


 classdef FNNetwork  
    properties
      Weights;  % vector of variables
      Error; 
      ErrorVec;
      Epocs;
      TrainData;
      TestData;
      Input;
      Target;
      
      Topology;
      NumLayers;
      Layer; 
      LOutput;
            
      Hidden;
      NumPattern;
            
       NumInput ;
       NumOutput ;
      
       NetOutput; % Output by Forward Pass
       
        HOutput ;
       % Predict ;
        TF ;
      
        Decomposition; % Layer based Decom (1), Network (2), NSP (3), Synapse (4)
         
       TrainClassPerf;
       TestClassPerf ;
       
       TrainRMSE;
       TestRMSE;
         
    end
   
   methods (Static)
        
       
       function obj = FNNetwork( TrainInput, TrainTarget, TestInput, TestTarget , Topology, Decom)
           obj.Decomposition = Decom;
           
           obj.TF =1;
           obj.Topology = Topology;
           obj.NumLayers = length(Topology);
           
           obj.NumInput = Topology(1);
           obj.NumOutput = Topology(end); 
           %obj.Hidden = Topology(2);      
           
           obj= FNNetwork.SetData(obj,TrainInput, TrainTarget, TestInput, TestTarget );
           obj =FNNetwork.SetNetwork(obj );
       end
       
        function W = GetWeights(obj)
         W = obj.AllWeights;
         end
      
        function E = GetError(obj)
         E = obj.SSE;
        end
        
        function Time = GetEpocs(obj)
         Time = obj.Epocs;
        end 
        
        function TrainPerf  = GetTrainClassPerf(obj) 
           TrainPerf  = obj.TrainClassPerf;
        end
     
        function TestPerf  =  GetTestClassPerf(obj)
           TestPerf =   obj.TestClassPerf ; 
        end
     
        function TrainRMSE = GetTrainRMSE(obj) 
           TrainRMSE  = obj.TrainRMSE;
        end
     
        function TestRMSE  =  GetTestRMSE(obj)
           TestRMSE =   obj.TestRMSE ; 
        end
     
        
        function obj = SetData(obj, TrainInput, TrainTarget, TestInput, TestTarget)  
          
          obj.TrainData.Input = TrainInput;  
          obj.TrainData.Target =    TrainTarget;
 
          obj.TestData.Input =   TestInput;
          obj.TestData.Target =    TestTarget;
%         
%         function obj = SetData(obj, TrainFile, TestFile) 
%           delimiterIn = ' ';
%           Data = importdata(TrainFile,delimiterIn);
%           delimiterIn = ' ';
%           TData = importdata(TestFile,delimiterIn);
%           obj.TrainData.Input = Data(1:end,1:end-obj.NumOutput);
%           obj.TrainData.Target =  Data(1:end,obj.NumInput+1:end);
% 
%          % obj.TrainData.Input  
%          % obj.TrainData.Target 
%           obj.TestData.Input = TData(1:end,1:end-obj.NumOutput);
%           obj.TestData.Target =  TData(1:end,obj.NumInput+1:end);
          
         obj.NumPattern  =  size(obj.TrainData.Input,1); % size of traindata
          
        end 
        function obj = SetNetwork(obj )
            
            
          for layer=1: obj.NumLayers-1 % traverse through the layers 
             obj.Layer(layer).Weights  = (randn(   obj.Topology(layer),  obj.Topology(layer+1)   ) - 0.5)/10;
             
          end
              for layer=2: obj.NumLayers % traverse through the layers  
             %obj.Layer(layer).Bias = (  rand(1,obj.Topology(layer)) - 0.5)/10; 
             obj.Layer(layer).Bias = 0.5;
          end
          
          %  for layer=1: obj.NumLayers-1 % traverse through the layers 
            % obj.Layer(layer).Weights   
           % end
           
             % for layer=2: obj.NumLayers % traverse through the layers  
            % obj.Layer(layer).Bias  
             % end
             
          obj.NetOutput = zeros(obj.NumPattern,obj.Topology(end)); % used to save the prediction by NN 
          %(used for training and test data. assumes that training data will be shorter than test data)
          
        end
        
         
         function TF = Trans(obj, x) 
             if obj.TF == 0
             TF = tanh(x);
             else
             TF =  1./(1+exp(-x));   
             end
         end 
         
         
         function E = RMSE(obj, Data)  % Mean Squarred Error 
              E =0;
            for patt =1:  size(Data.Input,1)  
                E = E + mean((obj.NetOutput(patt,:) - Data.Target(patt,:)).^2); 
            end  
           E = E /size(Data.Input,1) ;
               E= sqrt(E); 
         end
          
      function obj = ForwardPass(obj, p, Data) 
          
          obj.Layer(1).Output =  Data.Input(p,:); 
          obj.Target =  Data.Target(p,:);
          
          for layer =1:obj.NumLayers-1  
                    Sum = 0;
                  for row = 1: obj.Topology(layer+1)
                     for col = 1: obj.Topology(layer)
                         Sum = Sum + (obj.Layer(layer).Output(col) * obj.Layer(layer).Weights(col,row));  
                     end  
				     Forward = Sum - obj.Layer(layer+1).Bias(row); 
                     Sum = 0;
                    obj.Layer(layer+1).Output(row) = FNNetwork.Trans(obj,Forward) ; 
                  end   
          end 
          
          obj.NetOutput(p,:) =  obj.Layer(end).Output; %save
           
          
      end 
      
       function obj = BackwardPass(obj, LRate,p) 
           
           %NOTE: not sure if BP working - so we use EA only. There seems
           %to be some error in BP, so you can help me fix it
           
           Alpha = 0.01; % weight decay 
           
      for  x=1: obj.Topology(end)
		obj.Layer(end).Error(x) = (obj.Layer(end).Output(x)*(1- obj.Layer(end).Output(x)))*(obj.Target(x)-obj.Layer(end).Output(x) );
      end
      
      %obj.Layer(end).Error 
      %disp('Error');
       for layer = obj.NumLayers-1:-1:2 % go backwards 
           %layer
                    Sum = 0;
                  for row = 1: obj.Topology(layer)
                     for col = 1: obj.Topology(layer+1)
                         Sum = Sum + (obj.Layer(layer+1).Error(col) * obj.Layer(layer).Weights(row,col));  
                     end  
                    obj.Layer(layer).Error(row) = (obj.Layer(end).Output(x)*(1- obj.Layer(end).Output(x))) * Sum ;  
                    Sum = 0;
                  end   
                  
       % obj.Layer(layer).Error
                  
       end 
       % disp('Weightup');
       
       
       for layer = obj.NumLayers-1:-1:1 % go backwards  
          % layer
                  for row = 1: obj.Topology(layer)
                     for col = 1: obj.Topology(layer+1)  
              			tmp = ( LRate * obj.Layer(layer+1).Error(col) * obj.Layer(layer).Output(row)  );
     		        	obj.Layer(layer).Weights(row,col) = obj.Layer(layer).Weights(row,col) +  ( tmp  -  ( Alpha * tmp) ) ;%update weight
                     end    
                  end  
                
      % obj.Layer(layer).Weights  
                  
       end   
       % disp('Biasup');
       for layer = obj.NumLayers:-1:2 % go backwards  
         %  layer
                  for row = 1: obj.Topology(layer)  
              			tmp = ( -1 * LRate * obj.Layer(layer).Error(row)   );
     		        	 obj.Layer(layer).Bias(row) = obj.Layer(layer).Bias(row) +   ( tmp  -  ( Alpha * tmp) ) ;%update weight
                         
                  end   
                 % obj.Layer(layer).Bias 
                  
       end 
       
        
       end 
      
       function obj = SaveTrainedNet(obj, Solution,  Topo,    sp) 
           
          % if sp ==1
             obj = FNNetwork.TransSolutionWeights(obj,Solution,    Topo,   sp);
          % end
        %   if sp ==2
           %  obj = FNNetwork.TransSolutionWeights(obj,Solution,   T1Topo, T2Topo, sp);
          % end
           
       end
       
       function obj = TransSolutionWeights(obj,SolutionVec,  Topo,   sp)  % for EA - to encode EA solution in NN
           
           Top1  = Topo{1};
%            Top2  = Topo{2};
%            Top3       = Topo{3};
%            Top4       = Topo{4};
%            
           
           
           position = 1;
           
           
           
           if obj.Decomposition == 5 % Multitask
              %   obj.Topology
%                 In1 = SmallTopo(1);
%                 In1 = BiggerTopo(1);
%                 
%                 In3 = obj.Topology(1);
%                 
%                 H1 =  Top1(2); 
%                 H2 =  Top2(2);
%                 H3=  Top3(2); 
%                 H4 = Top4(2);
%                 
%                 H5 =   obj.Topology(2); 
             
%                 
%                 Out1 = SmallTopo(3); 
%                 Out2 = BiggerTopo(3);
%                 Out3 =  obj.Topology(3);
                
              for layer = 1: obj.NumLayers-1
               
               for neu =1:  Top1(layer+1)   
                   for row =1:  Top1(layer)  
                         obj.Layer(layer).Weights(row,neu)  = SolutionVec(position) ; 
                           position = position + 1;
                   end 
                           obj.Layer(layer+1).Bias(neu)  = SolutionVec(position) ; 
                            position = position + 1; 
               end  
               
             end 
             
          %   if In1 != In2 | H1 !=H2 | Out1 != Out2
            
%              %----------------   
%              
%              end
%              
%              
%              if  sp == 5
%                  
%                  MaxSteps = 5;
                 
%                  
%            Top1  = Topo{1};
%            Top2  = Topo{2};
%            Top3       = Topo{3};
%            Top4       = Topo{4};
%            
%                  
                 
                 
%                  
%                   H1 =  Top1(2); 
%                 H2 =  Top2(2);
%                 H3=  Top3(2); 
%                 H4 = Top4(2);
                 
          if sp >= 3 
                % sp
            for step = 2:sp-1
                %step
                 TopPrev = Topo{step-1};
                 TopG = Topo{step};
                   
                  Hid = TopPrev(2);
              
                   layer = 1;
             %  position
               for neu =Hid+1: TopG(layer+1)   
                %   neu
                   for row =1:  TopG(layer)  
                         obj.Layer(layer).Weights(row,neu)  = SolutionVec(position) ; 
                           position = position + 1;
                   end 
                           obj.Layer(layer+1).Bias(neu)  = SolutionVec(position) ; 
                            position = position + 1; 
               end  
               layer = 2;
               
               for neu = 1:  TopG(layer+1)   
                   for row =Hid+1:  TopG(layer)  
                         obj.Layer(layer).Weights(row,neu)  = SolutionVec(position) ; 
                           position = position + 1;
                   end 
                            
               end   
                % position
            end
            
          end
      %  ------------------------------------   
             
             % end part
             
             if sp >= 2
               layer = 1;
              
               TopGen = Topo{sp-1};
               
                HGen = TopGen(2) ;
                
                
               for neu =HGen+1: obj.Topology(layer+1)   
                   for row =1: obj.Topology(layer)  
                         obj.Layer(layer).Weights(row,neu)  = SolutionVec(position) ; 
                           position = position + 1;
                   end 
                           obj.Layer(layer+1).Bias(neu)  = SolutionVec(position) ; 
                            position = position + 1; 
               end  
                
               layer = 2;
               
               for neu = 1: obj.Topology(layer+1)   
                   for row =HGen+1: obj.Topology(layer)  
                         obj.Layer(layer).Weights(row,neu)  = SolutionVec(position) ; 
                           position = position + 1;
                   end 
                           %obj.Layer(layer+1).Bias(neu)  = SolutionVec(position) ; 
                           % position = position + 1; 
               end   
            
             
             end
 
    end %if decom is 5
           
           
           
           
       end
      
      function Er = EvaluateNNSol(obj,  SolutionVec, TaskTopo, sp) % Evaluate Encoded EA solution
          % map SolVec in Weights  
          
         Data =  obj.TrainData;
           
         obj =  FNNetwork.TransSolutionWeights(obj,SolutionVec, TaskTopo,  sp);   
         
            for p = 1:obj.NumPattern 
              obj=        FNNetwork.ForwardPass(obj,p, Data);  
            end 
          Er = FNNetwork.RMSE(obj, Data);
           
      end
      
      function ClassPerf = Count(obj, Data, ErTolerance) 
          
          Perf = 0;
          
           
         for patt = 1:size(Data.Input,1)  
             obj=   FNNetwork.ForwardPass(obj,patt, Data);  
         end
            
        
        for patt = 1:size(Data.Input,1)  
            
              T =  Data.Target(patt,:); 
              NetOut = obj.NetOutput(patt,:);
           
           for out = 1: obj.Topology(end)  % transform as per Er Tolerance (0.2 for training and 0.48 for test)
               
               if NetOut(out) <= ErTolerance
                   NetOut(out) = 0;
               else
               %if NetOut(out) >= (1- ErTolerance)
                   NetOut(out) = 1;
               end
                
           end  
           
            if isequal (T, NetOut)  
             Perf = Perf +1;   
            end
          
        end 
          
        
          ClassPerf = (Perf/size(Data.Input,1)) * 100;
      end
      
   function obj = TestClassificationNetwork(obj ) %   
     
     obj.TrainClassPerf = FNNetwork.Count(obj,  obj.TrainData, 0.2); 
     
     obj.TestClassPerf = FNNetwork.Count(obj,  obj.TestData, 0.48);
       
      end 
      
      
      
   function obj = TestRegressionNetwork(obj ) %   for time series and regressiion problems. will also give vale for classification
     
         for patt = 1:size(obj.TrainData.Input,1)  
             obj=   FNNetwork.ForwardPass(obj,patt, obj.TrainData);  
             
            % obj.TrainData.Target(patt, 1:end)
            % obj.NetOutput(patt, 1:end)
             
             
             % can output to file here - for time series problem - each
             % prediction
         end
         
     obj.TrainRMSE = FNNetwork.RMSE(obj,  obj.TrainData);  
     
         for patt = 1:size(obj.TestData.Input,1)  
             obj=   FNNetwork.ForwardPass(obj,patt, obj.TestData);  
         end
     
     
     obj.TestRMSE = FNNetwork.RMSE(obj,  obj.TestData ); 
      end 
      
      
      
      function obj = BP(obj, LRate, MaxEpocs)
       
           EVec = zeros(MaxEpocs,1)';
           
           Data =  obj.TrainData;
           
          for iter = 1:MaxEpocs
             iter
           
            for p = 1:obj.NumPattern 
               % p
              obj=        FNNetwork.ForwardPass(obj,p, Data);
               obj = FNNetwork.BackwardPass(obj, 0.1,p);
                 
            end
            
             EVec(iter) = FNNetwork.RMSE(obj);
            
            if Er < 0.01
             fprintf('converged at epoch: %d\n',iter);
              break 
            end 
         
          end 
          
       obj = FNNetwork.TestClassificationNetwork(obj ) ; 
     
          
        figure % opens new figure window
       plot(1:MaxEpocs,EVec);
           
      end 
   end
 end
       
