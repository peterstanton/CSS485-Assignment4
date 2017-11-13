function output = backprop()

 in0 = [0,1,1,1,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0];
 in1 = [0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
 in2 = [1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1];
 
 LearningRate = 0.1;
  
 t0 = [1,0,0];
 t1 = [0,1,0];
 t2 = [0,0,1];
 
 Input = [in0; in1; in2]';
 Output = [t0; t1; t2]';
 
 NumHidLayerNeurons = 2;
 
HiddenLayerWeights = rand(NumHidLayerNeurons,30); % Weight matrix from Input to Hidden
OutputLayerWeights = rand(3,NumHidLayerNeurons); % Weight matrix from Hidden to Output
biasHidden = rand(NumHidLayerNeurons,1);         % Random bias.
biasOutput = rand(3,1);
 
IterationCount = 0;  %Count passes.
ErrorVec(1:1000000) = 0;
xVec(1:1000000) = 0;
tester = 10.00;

while (tester > 0.05)
    i = randi(3);
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
 %  S2 = -2. * diag(ones(size(outOfOutput))-(outOfOutput.*outOfOutput)) * myError;
  
   S2 = -2.*diag((ones(size(outOfOutput))-outOfOutput).*outOfOutput)*myError;
   
 %  S1 = diag(ones(size(outOfHidden))-(outOfHidden.*outOfHidden)) * OutputLayerWeights'*S2;
   
   S1 = diag((ones(size(outOfHidden))-outOfHidden).*outOfHidden)*OutputLayerWeights'*S2;
   
   IterationCount
   S1 
   S2
   myError
   
   newW2 = OutputLayerWeights-LearningRate*S2*(outOfHidden');
   OutputLayerWeights = newW2;
      
   newW1 = HiddenLayerWeights-LearningRate*S1*(Input(:,i)');
   HiddenLayerWeights = newW1;
   
   newB2 = biasOutput-LearningRate.*S2;
   biasOutput = newB2;
   
   newB1 = biasHidden-LearningRate.*S1;
   biasHidden = newB1;
   xVec(IterationCount) = IterationCount;
   ErrorVec(IterationCount) = sum(myError.^2);
   tester = sum(myError.^2);
    
end

plot(xVec,ErrorVec)
title('Backpropagation Network Training')
xlabel('Backpropagation Iterations')
ylabel('Squared Error')