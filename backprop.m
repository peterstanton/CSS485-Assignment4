function output = backprop()

 in0 = [0,1,1,1,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0];
 in1 = [0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
 in2 = [1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1];
 w   = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
 b   = 0;
 
 HidLayerNeurons = 6;  % Number of hidden layer neurons
 
 t0 = [1,0,0];
 t1 = [0,1,0];
 t2 = [0,0,1];
 
 Input = [in0; in1; in2];
 Output = [t0; t1; t2];
 
HiddenLayerWeights = rand(3,3); % Weight matrix from Input to Hidden
 
OutputLayerWeights = rand(HidLayerNeurons,3); % Weight matrix from Hidden to Output
 
 
IterationCount = 0;  %Count passes.



function trainNet(Input,Output,HiddenLayerWeights,OutputLayerWeights)
 
 


