import CSV
import DataFrames
import FillArrays

function OLS(y, X)
    X = convert(Matrix, X)
    X_prime = X'
    inverted_Xs = inv(X_prime * X)
    beta = inverted_Xs * X_prime * y
    fitted = X * beta
    return beta, fitted
end


function Ridge(y, X, alpha)
    X = convert(Matrix, X)
    X_prime = X'
    I = FillArrays.Eye{Int}(size(X)[2],size(X)[2])
    inverted_Xs = inv(X_prime * X + alpha*I)
    beta = inverted_Xs * X_prime * y
    fitted = X * beta
    return beta, fitted
end
