module Polynomials

"""Evaluate linear comb. of Legendre polynomials given by c in points x."""
function legval(x::Array{Float64}, c::Array{Float64})
  if length(c) == 1
     c0 = c[1]
     c1 = 0
  elseif length(c) == 2
      c0 = c[1]
      c1 = c[2]
  else
    nd = length(c)
    c0 = c[end-1]
    c1 = c[end]

    for i in range(2, nd-2)
      tmp = c0
      nd = nd - 1
      c0 = c[end-i] - (c1*(nd - 1))/nd
      c1 = tmp + (c1.*x*(2*nd - 1))/nd
    end
  
  end

  return c0 + c1.*x
end

"""Compute m-th derivative of Legendre series as another Legendre series."""
function legder(c::Array{Float64}, m::Int64=1)
  # 0 order derivative
  if m == 0
    return c
  # len(c) is 0, 1, 2, 3 degree. Check for derivative of order larger than max
  # degree
  elseif m >= length(c)
    return Float64[]
  # Compute 
  else
    n = length(c)
    for i in range(1, m)
      n = n - 1
      der = zeros(n)
      c = copy(c)
     
      for j in n:-1:3
        der[j] = (2*j - 1)*c[j+1]
        c[j - 1] += c[j+1]
      end
      if n > 1
        der[2] = 3*c[3]
      end
      der[1] = c[2]
      c = der
    end
  end
  return c
end

end
