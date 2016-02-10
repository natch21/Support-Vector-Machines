function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

min_error = 1000;

vec = [0.01 0.03 0.1 0.3 1 10 30];
for i = vec
    for j = vec
        model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < min_error
            C = i;
            sigma = j;
            min_error = error;
        end
    end
end



end
