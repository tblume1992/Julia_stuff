import CSV
import DataFrames
import FillArrays

function OLS(y, X)
    X = convert(Matrix, X)
    X_prime = X'
    inverted_Xs = inv(X_prime * X)
    return beta = inverted_Xs * X_prime * y
    return fitted = X * beta
end


function Ridge(y, X, alpha)
    X = convert(Matrix, X)
    X_prime = X'
    I = FillArrays.Eye{Int}(size(X)[2],size(X)[2])
    inverted_Xs = inv(X_prime * X + alpha*I)
    return beta = inverted_Xs * X_prime * y
    return fitted = X * beta
end
