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
 
 NumHidLayerNeurons = 5;
 
HiddenLayerWeights = rand(NumHidLayerNeurons,30); % Weight matrix from Input to Hidden
OutputLayerWeights = rand(3,NumHidLayerNeurons); % Weight matrix from Hidden to Output
biasHidden = rand(NumHidLayerNeurons,1);         % Random bias.
biasOutput = rand(3,1);
 
IterationCount = 0;  %Count passes.
ErrorVec1(1:1000000) = 0;
xVec(1:1000000) = 0;
tester = 10.00;

%while (tester > 0.05)
while(IterationCount < 10000)
    i = randi(3);
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
  
   S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
   S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;
         
   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec1(IterationCount) = sum(myError.^2)/length(myError);
   tester = sum(myError.^2);    
end


IterationCount = 0;  %Count passes.
ErrorVec0(1:1000000) = 0;
xVec(1:1000000) = 0;
tester = 10.00;

%while (tester > 0.05)
while(IterationCount < 10000)
    i = randi(3);
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
  
   S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
   S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;
         
   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec0(IterationCount) = sum(myError.^2)/length(myError);
   tester = sum(myError.^2);    
end

%E = [ErrorVec(1:end-3) + ErrorVec(2:end-2) + ErrorVec(3:end-1)]/3;

corrupt1 = randi([1 30]);
corrupt2 = randi([1 30]);
corrupt3 = randi([1 30]);
corrupt4 = randi([1 30]);

for i = 1:3
    Input(corrupt1,i) = ~Input(corrupt1,i);
    Input(corrupt2,i) = ~Input(corrupt2,i);
    Input(corrupt3,i) = ~Input(corrupt3,i);
    Input(corrupt4,i) = ~Input(corrupt4,i);
end


ErrorVec2(1:1000000) = 0;
IterationCount = 0;
tester = 10.0;
myError = 0;

%while (tester > 0.05)
while(IterationCount < 5000)
    i = randi(3);
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
  
   S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
   S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;
         
   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec2(IterationCount) = sum(myError.^2)/length(myError);
   tester = sum(myError.^2);    
end


corrupt1 = randi([1 30]);
corrupt2 = randi([1 30]);
corrupt3 = randi([1 30]);
corrupt4 = randi([1 30]);

corrupt5 = randi([1 30]);
corrupt6 = randi([1 30]);
corrupt7 = randi([1 30]);
corrupt8 = randi([1 30]);


for i = 1:3
    Input(corrupt1,i) = ~Input(corrupt1,i);
    Input(corrupt2,i) = ~Input(corrupt2,i);
    Input(corrupt3,i) = ~Input(corrupt3,i);
    Input(corrupt4,i) = ~Input(corrupt4,i);
    Input(corrupt5,i) = ~Input(corrupt5,i);
    Input(corrupt6,i) = ~Input(corrupt6,i);
    Input(corrupt7,i) = ~Input(corrupt7,i);
    Input(corrupt8,i) = ~Input(corrupt8,i);
end


ErrorVec3(1:1000000) = 0;
IterationCount = 0;
tester = 10.0;
myError = 0;

while(IterationCount < 5000)
    i = randi(3);
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
  
   S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
   S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;
         
   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec3(IterationCount) = sum(myError.^2)/length(myError);
   tester = sum(myError.^2);    
end

figure(1)
plot(xVec,ErrorVec1,xVec,ErrorVec2,xVec,ErrorVec3)
title('Backpropagation Network Training')
xlabel('Backpropagation Iterations')
ylabel('Squared Error')


graphx = 0:4:8;
graphy = [mean(ErrorVec0) mean(ErrorVec2) mean(ErrorVec3)];
figure(2)
bar(graphx,graphy);
title('Mean Error for Backpropagation Learning Digit Recognition')
xlabel('Corrupted Pixels')
ylabel('Mean Error Rate')
