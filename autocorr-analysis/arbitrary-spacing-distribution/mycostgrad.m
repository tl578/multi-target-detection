function [f, G] = mycostgrad(z, params)

    L = params.L ;
    ay1 = params.ay1 ;
    ay2 = params.ay2 ;
    ay3 = params.ay3 ;
    sigma = params.sigma ;
    nbin = params.nbin ;

    w1 = 1 ;
    w2 = 1 / nbin ;
    w3 = 2 / nbin*(nbin+1) ;
    bsz = L/nbin ;

    %% redistribute ay2
    ay2(1) = ay2(1) - sigma^2 ;
    new_ay2 = zeros(nbin, 1) ;
    count = zeros(nbin, 1) ;
    for i = 1:L
        i0 = round(i/bsz - 0.5 - 1e-6) + 1 ;
        if (i0 > nbin)
            i0 = nbin ;
        end
        new_ay2(i0) = new_ay2(i0) + ay2(i) ;
        count(i0) = count(i0) + 1 ;
    end

    for i = 1:nbin
        new_ay2(i) = new_ay2(i)/count(i) ;
    end
    ay2 = new_ay2 ;

    %% redistribute ay3
    for l2 = 1:L
        for l1 = 1:l2
            if (l1 == 1)
                ay3(l1, l2) = ay3(l1, l2) - sigma^2*ay1 ;
            end
            if (l2 == 1)
                ay3(l1, l2) = ay3(l1, l2) - sigma^2*ay1 ;
            end
            if (l1 == l2)
                ay3(l1, l2) = ay3(l1, l2) - sigma^2*ay1 ;
            end
        end
    end

    new_ay3 = zeros(nbin, nbin) ;
    count = zeros(nbin, nbin) ;
    for j = 1:L
        j0 = round(j/bsz - 0.5 - 1e-6) + 1 ;
        if (j0 > nbin)
            j0 = nbin ;
        end

        for i = 1:j
            i0 = round(i/bsz - 0.5 - 1e-6) + 1 ;
            if (i0 > nbin)
                i0 = nbin ;
            end
            new_ay3(i0, j0) = new_ay3(i0, j0) + ay3(i, j) ;
            count(i0, j0) = count(i0, j0) + 1 ;
        end
    end

    for j = 1:nbin
        for i = 1:j
            new_ay3(i, j) = new_ay3(i, j)/count(i, j) ;
        end
    end
    ay3 = new_ay3 ;

    %% construct low-resolution signal
    fx = zeros(L, 1) ;
    fx(1) = z.x(1) ;
    if (nbin < L)
        imax = nbin/2 ;
    elseif (rem(nbin, 2) == 0)
        imax = nbin/2 - 1 ;
        fx(nbin/2 + 1) = z.x(nbin) ;
    else
        imax = (nbin-1)/2 ;
    end

    for i = 1:imax
        a = z.x(2*i) ;
        b = z.x(2*i+1) ;
        fx(i+1) = a + b*1j ;
        fx(L+1-i) = a - b*1j ;
    end
    
    x_est = zeros(2*L, 1) ;
    x_est(1:L) = real(ifft(fx)) ;
    rho = z.rho ;
    g_x = zeros(size(z.x, 1), 1) ;
    g_rho = zeros(size(z.rho, 1), 1) ;


    %% first order
    ax1_est = sum(x_est) ;
    ay1_est = rho(1)*ax1_est ;
    f = w1*(ay1 - ay1_est)^2 ;

    coeff = zeros(size(z.x, 1), 1) ;
    fc = zeros(L, 1) ;
    coeff(1) = 1 ;
    if (nbin == L && rem(nbin, 2) == 0)
        coeff(nbin) = 0 ;
    end

    for i = 1:imax
        fc(i+1) = 1 ;
        fc(L+1-i) = 1 ;
        coeff(2*i) = real(sum(ifft(fc))) ;

        fc(i+1) = 1j ;
        fc(L+1-i) = -1j ;
        coeff(2*i+1) = real(sum(ifft(fc))) ;

        fc(i+1) = 0 ;
        fc(L+1-i) = 0 ;
    end

    g_x = -2*w1*(ay1 - ay1_est)*rho(1)*coeff ;
    g_rho(1) = -2*w1*(ay1 - ay1_est)*ax1_est ;


    %% second order
    ax2_est = zeros(nbin, 1) ;
    for i = 1:nbin
        i0 = round((i-1)*bsz + 0.5) ;
        ax2_est(i) = sum(x_est(1:L) .* x_est(i0:L+i0-1)) ;
    end

    ay2_est = rho(1)*ax2_est ;
    for i = 2:nbin
        for l = 2:i
            ay2_est(i) = ay2_est(i) + rho(l)*ax2_est(nbin-i+l) ;
        end
    end
    
    f = f + w2*sum((ay2 - ay2_est).^2) ;

    gx_ay2_est = zeros(nbin, L) ;
    for k = 1:L
        for l = 1:nbin
            val = 0 ;
            l0 = round((l-1)*bsz + 0.5) ;
            idx = k + l0 - 1 ;
            if (idx > 0 && idx <= L)
                val = val + rho(1)*x_est(idx) ;
            end ;
            idx = k - l0 + 1 ;
            if (idx > 0 && idx <= L)
                val = val + rho(1)*x_est(idx) ;
            end
            
            for i = 2:l
                i0 = round((nbin-l+i-1)*bsz + 0.5) ;
                idx = k + i0 - 1 ;
                if (idx > 0 && idx <= L)
                    val = val + rho(i)*x_est(idx) ;
                end
                idx = k - i0 + 1 ;
                if (idx > 0 && idx <= L)
                    val = val + rho(i)*x_est(idx) ;
                end
            end
            gx_ay2_est(l, k) = val ;
        end
    end

    diff_ay2 = ay2 - ay2_est ;
    for l = 1:nbin
        val = -2*w2*diff_ay2(l)/L ;
        fc = fft(gx_ay2_est(l, :)) ;
        g_x(1) = g_x(1) + val*real(fc(1)) ;
        if (nbin == L && rem(nbin, 2) == 0)
            g_x(nbin) = g_x(nbin) + val*real(fc(nbin/2 + 1)) ;
        end

        for i = 1:imax
            g_x(2*i) = g_x(2*i) + val*real(fc(i+1) + fc(L+1-i)) ;
            g_x(2*i+1) = g_x(2*i+1) + val*imag(fc(i+1) - fc(L+1-i)) ;
        end
    end

    g_rho(1) = g_rho(1) - 2*w2*sum(diff_ay2 .* ax2_est) ;
    for k = 2:nbin
        grho_ay2_est = zeros(nbin, 1) ;
        for l = k:nbin
            grho_ay2_est(l) = ax2_est(nbin-l+k) ;
        end
        g_rho(k) = g_rho(k) - 2*w2*sum(diff_ay2 .* grho_ay2_est) ;
    end


    %% third order
    ax3_est = zeros(nbin, nbin) ;
    for j = 1:nbin
        j0 = round((j-1)*bsz + 0.5) ;
        for i = 1:j
            i0 = round((i-1)*bsz + 0.5) ;
            ax3_est(i, j) = sum((x_est(1:L) .* x_est(i0:L+i0-1)) .* x_est(j0:L+j0-1)) ;
        end
    end

    ay3_est = rho(1)*ax3_est ;
    for l2 = 1:nbin
        for l1 = 1:l2
            for i = 1:l2-l1
                ay3_est(l1, l2) = ay3_est(l1, l2) + rho(i+1)*ax3_est(nbin-l2+i+1, nbin-(l2-l1)+i) ;
            end
            for i = 1:l1-1
                ay3_est(l1, l2) = ay3_est(l1, l2) + rho(i+1)*ax3_est(l2-l1+1, nbin-l1+i+1) ;
            end
        end
    end

    diff_ay3 = ay3 - ay3_est ; 
    for l2 = 1:nbin
        for l1 = 1:l2
            f = f + w3*diff_ay3(l1, l2)^2 ;
        end
    end

    gx_ay3_est = zeros(nbin, nbin, L) ;
    for k = 1:L
        for l2 = 1:nbin
            for l1 = 1:l2
                val = 0 ;
                i0 = round((l1-1)*bsz + 0.5) ;
                j0 = round((l2-1)*bsz + 0.5) ;
                idx1 = k + (i0 - 1) ;
                idx2 = k + (j0 - 1) ;
                if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                    val = val + x_est(idx1)*x_est(idx2) ;
                end

                idx1 = k - (i0 - 1) ;
                idx2 = k + (j0 - i0) ;
                if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                    val = val + x_est(idx1)*x_est(idx2) ;
                end

                idx1 = k - (j0 - 1) ;
                idx2 = k - (j0 - i0) ;
                if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                    val = val + x_est(idx1)*x_est(idx2) ;
                end
                gx_ay3_est(l1, l2, k) = rho(1)*val ;

                for i = 1:l2-l1
                    val = 0 ;
                    i0 = round((nbin-l2+i)*bsz + 0.5) ;
                    j0 = round((nbin-(l2-l1)+i-1)*bsz + 0.5) ;
                    idx1 = k + (i0 - 1) ;
                    idx2 = k + (j0 - 1) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end

                    idx1 = k - (i0 - 1) ;
                    idx2 = k + (j0 - i0) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end
                    
                    idx1 = k - (j0 - 1) ;
                    idx2 = k - (j0 - i0) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end

                    gx_ay3_est(l1, l2, k) = gx_ay3_est(l1, l2, k) + rho(i+1)*val ;
                end

                for i = 1:l1-1
                    val = 0 ;
                    i0 = round((l2-l1)*bsz + 0.5) ;
                    j0 = round((nbin-l1+i)*bsz + 0.5) ;
                    idx1 = k + (i0 - 1) ;
                    idx2 = k + (j0 - 1) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end
                    
                    idx1 = k - (i0 - 1) ;
                    idx2 = k + (j0 - i0) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end
                    
                    idx1 = k - (j0 - 1) ;
                    idx2 = k - (j0 - i0) ;
                    if (idx1 > 0 && idx1 <= L && idx2 > 0 && idx2 <= L)
                        val = val + x_est(idx1)*x_est(idx2) ;
                    end

                    gx_ay3_est(l1, l2, k) = gx_ay3_est(l1, l2, k) + rho(i+1)*val ;
                end
            end
        end
    end

    gx_ay3_est_l1_l2 = zeros(L, 1) ;
    for l2 = 1:nbin
        for l1 = 1:l2
            val = -2*w3*diff_ay3(l1, l2)/L ;
            for k = 1:L
                gx_ay3_est_l1_l2(k) = gx_ay3_est(l1, l2, k) ;
            end
            fc = fft(gx_ay3_est_l1_l2) ;

            g_x(1) = g_x(1) + val*real(fc(1)) ;
            if (nbin == L && rem(nbin, 2) == 0)
                g_x(nbin) = g_x(nbin) + val*real(fc(nbin/2 + 1)) ;
            end

            for i = 1:imax
                g_x(2*i) = g_x(2*i) + val*real(fc(i+1) + fc(L+1-i)) ;
                g_x(2*i+1) = g_x(2*i+1) + val*imag(fc(i+1) - fc(L+1-i)) ;
            end
        end
    end
    
    for l2 = 1:nbin
        for l1 = 1:l2
            g_rho(1) = g_rho(1) - 2*w3*diff_ay3(l1, l2)*ax3_est(l1, l2) ;
        end
    end

    for k = 2:nbin
        grho_ay3_est = zeros(nbin, nbin) ;
        for l2 = 1:nbin
            for l1 = 1:l2
                val = 0 ;
                if (k <= l2 - l1 + 1)
                    val = val + ax3_est(nbin+k-l2, nbin+k+l1-l2-1) ;
                end
                if (k <= l1)
                    val = val + ax3_est(l2-l1+1, nbin+k-l1) ;
                end
                grho_ay3_est(l1, l2) = val ;
            end
        end
        
        for l2 = 1:nbin
            for l1 = 1:l2
                g_rho(k) = g_rho(k) - 2*w3*diff_ay3(l1, l2)*grho_ay3_est(l1, l2) ;
            end
        end
    end

    G = struct() ;
    G.x = g_x ;
    G.rho = g_rho ;

end
