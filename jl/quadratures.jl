module Quadratures

"""Points and weight for exact integration of polynomials of degree = 2*degree -1."""
function gauss_legendre(degree::Int)
  #=
  The implementation is based on eigenvalues of the matrix T that is similar to
  the Jacobi matrix J. Both are tridiagonal but the T matrix is symmetric and
  the main diagonal is 0.
  =#
  diag = zeros(degree)
  up_diag = Float64[k/sqrt(4*k^2-1) for k=1:degree-1]
  T = SymTridiagonal(diag, up_diag)
  xq, ev = eig(T)
  wq = reshape(2*ev[1, :].^2, degree)
  return xq, wq
end

end
