
function [Dimen, D] = SetCCNN(Topology,  H) 
            
             W1 = Topology(1)*Topology(2); % W1
             W2 = Topology(2)  * Topology(3); %W2
             B = Topology(2)  + Topology(3); % Bias   
            % B = Topology(2) ; % Bias   
             D = W1+W2 +B;  % total dimension  of smallest network for 1st SP
            
             Dimen(1) = D; %first SP
                
               for n = 1: length(H)-1 %rest of SP 
                   Hdiff = H(n+1) - H(n); 
                    Dimen(n+1) = (Topology(1) +1 + Topology(3)) *  Hdiff; %input links + bias + output links 
               end   
               
               Dimen
end
