function output = backprop()

 in0 = [0,1,1,1,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0];
 in1 = [0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
 in2 = [1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1];
 w   = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
 
 
 HidLayerNeurons = 6;  % Number of hidden layer neurons
 
 t0 = [1,0,0];
 t1 = [0,1,0];
 t2 = [0,0,1];
 
 Input = [in0; in1; in2];
 Output = [t0; t1; t2];
 
HiddenLayerWeights = rand(3,3); % Weight matrix from Input to Hidden
 
OutputLayerWeights = rand(HidLayerNeurons,3); % Weight matrix from Hidden to Output

biasHidden = rand(HidLayerNeurons,3);         % Random bias.
 
IterationCount = 0;  %Count passes.

error = 1.00;

while (error > 0.05)
    IterationCount = IterationCount + 1;                %Increment the counter
    Error_Mat(IterationCount)=error;                    %Put error rate in matrix.
    outOfHidden = Input;                                %start computing output
    outOfHidden = HiddenLayerWeights' * outOfHidden;    %Mult by weight matrix.
    outOfHidden = outOfHidden + biasHidden;             %Add bias
    outOfHidden = logsig(outOfHidden);                  %Apply log sigmoid function
    
    
    outOfOutput = outOfHidden;
    outOfOutput = OutputLayerWeights' * outOfOutput;
    outOfOutput = outOfOutput + biasOutput;
    outOfOutput = logsig(outOfOutput);
    
    myError = Output - outOfOutput;
    
    S2 = -2 * diff(logsig(n^2 * e
    
        
