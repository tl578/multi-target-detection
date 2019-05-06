function [] = main(my_rand_seed, autocorr_file, outfile)

    cur_dir = pwd ;
    cd '~/MATLAB/manopt' ;
    importmanopt ;
    eval(['cd ', cur_dir]) ;
    
    my_rand_seed = str2num(my_rand_seed) ;
    rng(my_rand_seed) ;

    % read in autocorrelations
    fp = fopen(autocorr_file, 'r') ;
    L = fscanf(fp, '%d', 1) ;
    sigma = fscanf(fp, '%lf', 1) ;
    ay1 = fscanf(fp, '%lf', 1) ;
    ay2 = fscanf(fp, '%lf', L) ;
    ay3 = zeros(L, L) ;
    for l2 = 1:L
        for l1 = 1:l2
            ay3(l1, l2) = fscanf(fp, '%lf', 1) ;
        end
    end
    fclose(fp) ;
    
    params.L = L ;
    params.sigma = sigma ;
    params.ay1 = ay1 ;
    params.ay2 = ay2 ;
    params.ay3 = ay3 ;

    if (rem(L, 2) == 0)
        jmax = L/2 ;
    else
        jmax = (L+1)/2 ;
    end
    
    sol_x = zeros(L, jmax) ;
    sol_rho = zeros(L, jmax) ;
    for j = 1:jmax
        nbin = 2*j ;
        params.nbin = nbin ;
        xlen = nbin + 1 ;
        if (nbin >= L)
            nbin = L ;
            xlen = L ;
        end
    
        if (j == 1)
            x0 = randn(xlen, 1) ;
        else
            x0 = zeros(xlen, 1) ;
            fx = fft(x_est) ;
            x0(1) = real(fx(1)) ;
            if (nbin < L)
                imax = nbin/2 ;
            elseif (rem(nbin, 2) == 0)
                imax = nbin/2 - 1 ;
                x0(nbin) = real(fx(nbin/2 + 1)) ;
            else
                imax = (nbin-1)/2 ;
            end
            
            for i = 1:imax
                x0(2*i) = 0.5*real(fx(i+1) + fx(L+1-i)) ;
                x0(2*i+1) = 0.5*imag(fx(i+1) - fx(L+1-i)) ;
            end
        end
    
        if (j == 1)
            rho0 = abs(randn(nbin, 1)) ;
        else
            rho0 = zeros(nbin, 1) ;
            len = size(rho_est, 1) ;
    
            s = 0 ;
            for i = 2:len
                s = s + rho_est(i) ;
            end
            
            s = s/(nbin-1) ;
            rho0(1) = rho_est(1) ;
            for i = 2:nbin
                rho0(i) = s ;
            end
        end
    
        z0.x = x0 ;
        z0.rho = rho0 ;
        [x_est, rho_est, stats] = inv_autocorr(params, z0) ;
    
        for i = 1:L
            sol_x(i, j) = x_est(i) ;
        end
    
        for i = 1:nbin
            sol_rho(i, j) = rho_est(i) ;
        end
    
    end

    save(outfile) ;
    exit ;
end
