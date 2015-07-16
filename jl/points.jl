module Points

using Quadratures

"""Degree + 1 points evenly distributed on (-1, 1). Define polynomial of degree."""
function equidistant_points(degree::Int64)
  if degree == 0
    return [0.]
  elseif degree == 1
    return [-1., 1.]
  else
    return linspace(-1, 1, degree + 1)
  end
end


"""Degree + 1 points Chebyshev points distributed on (-1, 1). Define polynomial of degree."""
function chebyshev_points(degree::Int64)
  if degree == 0
    return [0.]
  elseif degree == 1
    return [-1., 1.]
  else
    return union([-1.], [-cos(pi*(2*k-1)/(2*(degree + 1))) for k in 2:degree], [1.])
  end
end


"""Degree + 1 points Gauss-Legendre points distributed on (-1, 1). Define polynomial of degree."""
function gauss_legendre_points(degree::Int64)
  if degree == 0
    return [0.]
  elseif degree == 1
    return [-1., 1.]
  else
    return union([-1.], Quadratures.gauss_legendre(degree-1)[1], [1.])
  end
end

# Add Gauss-Legendre nodes

end
