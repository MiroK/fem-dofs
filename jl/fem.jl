include("./polynomials.jl")
using Polynomials

function get_basis(deg::Int64)
    # Legendre polynomials are used as the basis of polynomials. In the basis of
    # Legendre polynomials is row of eye
    dim = deg + 1
    polyset = eye(dim)
   
    # Rearange dofs to have external first
    dofset = linspace(-1, 1, dim)
    dofset = [dofset[1]; dofset[end]; dofset[2:end-1]]
    # Compute the nodal matrix
    A = zeros(dim, dim)
    for col in 1:dim
      A[col, :] = Polynomials.legval(dofset, polyset[:, col])
    end

    # New coefficients
    B = inv(A) 

    # Combine the basis according to new weights
    function eval_basis(x::Array{Float64}, i::Int64)
      @assert 1 <= i <= dim
      Polynomials.legval(x, B[i, :])
    end

    # Check that the basis is nodal
    foo =  [eval_basis(dofset, i) for i in 1:dim]
    println(foo)

    # First deriv
    # dbasis = [lambda x, c=legder(c): legval(x, c) for c in B]
    
    return eval_basis
end

get_basis(4)
