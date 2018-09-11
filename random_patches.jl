import CSV
import DataFrames
import FillArrays
import Random
import StatsBase
using Missings



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
function RandomSubspaces(X, subspaces)
    random_X = X[:, filter(x -> (x in collect(1:DataFrames.length(X))),
                    Random.randperm(length(X)))]
    random_X = random_X[:,1:subspaces]
    return random_X
end

function RandomPatches(y, X, num_it, subspaces, sample = .7)
    coefficients = DataFrames.DataFrame(zeros(size(X)[2]),names(X))

    for i = 1:num_it

        random_X = X[:, filter(x -> (x in collect(1:DataFrames.length(X))),
                        Random.randperm(length(X)))]
        rs_X = random_X[:,1:subspaces]
        rs_X = hcat(rs_X,y)
        rp_X = rs_X[StatsBase.sample(1:DataFrames.nrow(rs_X),
                floor(Int,sample*DataFrames.nrow(rs_X))), :]
        y_fit = rp_X[:, end]
        delete!(rp_X, names(rp_X[:, end:end]))
        m_X = convert(Matrix, rp_X)
        X_prime = m_X'
        inverted_Xs = inv(X_prime * m_X)
        beta = inverted_Xs * X_prime * y_fit
        fitted = m_X * beta
        beta = DataFrames.DataFrame(beta, names(rp_X))
        for n in unique([names(coefficients); names(beta)]), df in [coefficients, beta]
        n in names(df) || (df[n] = missing)
        end
        coefficients = [coefficients; beta] 





    end
    coefficients = coefficients[setdiff(1:end, 1), :]
    return coefficients
end
