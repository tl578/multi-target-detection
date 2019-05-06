function [x_est, rho_est, stats] = inv_autocorr(params, z0)

    L = params.L ;
    ay1 = params.ay1 ;
    ay2 = params.ay2 ;
    ay3 = params.ay3 ;
    nbin = params.nbin ;

    xlen = nbin + 1 ;
    if (nbin == L)
        xlen = nbin ;
    end

    % elements.x stores the Fourier coeffients of the estimated signal
    elements.x = euclideanfactory(xlen) ;
    elements.rho = positivefactory(nbin) ;
    manifold = productmanifold(elements) ;
    
    problem.M = manifold ;
    problem.costgrad = @costgrad ;
        function [f, G] = costgrad(z)
            [f, G] = mycostgrad(z, params) ;
            G = manifold.egrad2rgrad(z, G) ;
        end

%    checkgradient(problem)

    opts = struct() ;
    opts.maxiter = 2000 ;
    if (nbin == L)
        opts.maxiter = 10000 ;
    end
    opts.tolgradnorm = 1e-6 ;

    warning('off', 'manopt:getHessian:approx') ;
    [z_est, loss, stats] = trustregions(problem, z0, opts) ;
    
    fx = zeros(L, 1) ;
    fx(1) = z_est.x(1) ;
    if (nbin < L)
        imax = nbin/2 ;
    elseif (rem(nbin, 2) == 0)
        imax = nbin/2 - 1 ;
        fx(nbin/2 + 1) = z_est.x(nbin) ;
    else
        imax = (nbin-1)/2 ;
    end

    for i = 1:imax
        a = z_est.x(2*i) ;
        b = z_est.x(2*i+1) ;
        fx(i+1) = a + b*1j ;
        fx(L+1-i) = a - b*1j ;
    end
    
    x_est = real(ifft(fx)) ;
    rho_est = z_est.rho ;
end
