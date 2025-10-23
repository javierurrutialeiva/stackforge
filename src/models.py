def gnfw(r, params):
    p0, gamma, beta, alpha, rs = params
    x = r/rs
    prof = p0/(x**gamma * (1 + x**alpha)**((beta - gamma)/alpha))
    return prof

def test_model(x, params):
    m, b = params
    y = m * x + b
    return y