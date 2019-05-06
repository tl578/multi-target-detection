#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <mpi.h>

const int root = 0 ;
const int num_class = 2 ;
const double golden_ratio = 1.618034 ;
char datafile[256] = "micrograph.bin" ;

int nproc, myid, L, trans_sz, (*shift_table)[2] ;
double *psf, *prev_psf, *grad, *search_vec ;
double **data, **prior, **log_prior ;
double *x_est, *prev_x_est, *inter_weight ;
double ***p_matrix, **psum_vec, *in ;
double **model_sqsum, *data_sqsum ;
double sigma, inv_var, inv_trans_sz, err, LL ;
long long Npix, num_data, s_num_data ;
fftw_complex *out, *F_est, **data_Fconj, *unity_F ;
fftw_plan forward_plan, backward_plan ;

void setup() ;
void freq_marchingEM() ;
void exp_max( int ) ;
void update_psf( int ) ;
double func_val( int, double ) ;
void redist_prior( int ) ;
void cal_likelihood( int ) ;
void free_mem() ;

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv) ;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid) ;

    double t1, t2 ;
    t1 = MPI_Wtime() ;
    setup() ;

    forward_plan = fftw_plan_dft_r2c_1d(trans_sz, in, out, FFTW_MEASURE) ;
    backward_plan = fftw_plan_dft_c2r_1d(trans_sz, out, in, FFTW_MEASURE) ;

    freq_marchingEM() ;

    fftw_destroy_plan(forward_plan) ;
    fftw_destroy_plan(backward_plan) ;
    free_mem() ;

    MPI_Barrier(MPI_COMM_WORLD) ;
    t2 = MPI_Wtime() ;
    if (myid == root)
        printf("total elapsed time = %.0f sec\n", t2-t1) ;

    MPI_Finalize() ;
    return 0 ;
}


void freq_marchingEM(){

    FILE *fp ;
    int k, m, n, r, s, n_max, num_step, sep_ct ;
    int idx, iter_ct, min_iter = 50, max_iter = 10000 ;
    double t1, t2 ;

    if (myid == root){
        fp = fopen("Data/output.dat", "w") ;
        fclose(fp) ;
    }

    n_max = floor((L+1)/2.) ;
    for (n = 1 ; n <= n_max ; n++){
        m = 2*n ;
        if (m > L)
            m = L ;
        num_step = 2*m ;
        sep_ct = 0 ;
        for (r = 1 ; r < num_step ; r++){
            for (s = r+1 ; s < num_step ; s++){
                if (s-r < num_step/2)
                    continue ;
                shift_table[sep_ct][0] = r ;
                shift_table[sep_ct][1] = s ;
                sep_ct += 1 ;
            }
        }

        /* initialize prior */
        if (n == 1){
            /* no or one particle */
            for (k = 0 ; k < num_step ; k++)
                prior[0][k] = 1./(num_step + 1) ;
            /* two particles */
            prior[1][0] = 1./(num_step + 1) ;
        }

        err = 1. ;
        t1 = MPI_Wtime() ;
        memcpy(prev_x_est, x_est, trans_sz*sizeof(double)) ;

        iter_ct = 0 ;
        if (num_step < trans_sz){
            while ((err > 2e-6 || iter_ct < min_iter) && iter_ct < max_iter){
                exp_max( num_step ) ;
                iter_ct += 1 ;
            }
        }
        else{
            while ((err > 1e-6 || iter_ct < min_iter) && iter_ct < max_iter){
                exp_max( num_step ) ;
                iter_ct += 1 ;
            }
        }
        
        if (myid == root){
            fp = fopen("Data/output.dat", "a") ;
            for (k = L ; k < trans_sz ; k++)
                fprintf(fp, "%lf ", x_est[k]) ;
            fprintf(fp, "\n") ;
            for (k = 0 ; k < num_step ; k++)
                fprintf(fp, "%lf ", prior[0][k]) ;
            fprintf(fp, "\n") ;
            for (k = 0 ; k < m+1 ; k++)
                fprintf(fp, "%lf ", psf[k]) ;
            fprintf(fp, "\n") ;
            fclose(fp) ;
        }

        if (num_step < trans_sz)
            redist_prior( num_step/2 ) ;
        else{
            cal_likelihood( num_step ) ;

            if (myid == root){
                fp = fopen("Data/output.dat", "a") ;
                fprintf(fp, "%1.10e\n", LL) ;
                fclose(fp) ;
            }
        }

        t2 = MPI_Wtime() ;
        if (myid == root){
            m += 2 ;
            if (m > L)
                m = L ;
            printf("\n") ;
            for (k = 0 ; k < m+1 ; k++)
                printf("%.7f ", psf[k]) ;
            printf("\nnum_step = %d: elapsed time = %.3f sec\n\n\n", num_step, t2-t1) ;
        }
    }
}


void exp_max( int num_step ){

    int i, d, m, r, s, idx, idx_start, num_byte ;
    int i0, j0, i1, j1, dmin, dmax, sep_ct ;
    double step_sz, L2, max_log_p, rescale ;
    double psum, inv_psum, f1, f2, ct ;

    num_byte = trans_sz*sizeof(double) ;
    memcpy(in, x_est, num_byte) ;
    step_sz = trans_sz / ((double) num_step) ;

    fftw_execute(forward_plan) ;
    for (i = 0 ; i < trans_sz ; i++){
        F_est[i][0] = out[i][0] ;
        F_est[i][1] = out[i][1] ;
    }

    /* no or one particle */
    for (r = 0 ; r < num_step ; r++)
        log_prior[0][r] = log(prior[0][r]) ;
    
    for (r = 0 ; r < num_step ; r++){
        model_sqsum[0][r] = 0. ;
        idx_start = round(r*step_sz) ;
        for (i = 0 ; i < L ; i++){
            idx = (idx_start + i) % trans_sz ;
            model_sqsum[0][r] += x_est[idx]*x_est[idx] ;
        }
    }

    /* two particles */
    sep_ct = (num_step/2)*(num_step/2 - 1)/2 ;
    for (r = 0 ; r < sep_ct ; r++)
        log_prior[1][r] = log(prior[1][r]) ;

    for (r = 0 ; r < sep_ct ; r++){
        model_sqsum[1][r] = 0. ;
        i0 = round(shift_table[r][0]*step_sz) ;
        j0 = round(shift_table[r][1]*step_sz) ;
        for (s = 0 ; s < L ; s++){
            i1 = (i0 + s) % trans_sz ;
            j1 = (j0 + s) % trans_sz ;
            model_sqsum[1][r] += pow(x_est[i1] + x_est[j1], 2) ;
        }
    }

    dmin = 0 ;
    dmax = s_num_data ;
    if (myid == nproc-1)
        dmax = num_data - (nproc-1)*s_num_data ;

    /* E-step */
    f1 = 2*inv_trans_sz ;
    f2 = 0.5*inv_var ;
    for (d = dmin ; d < dmax ; d++){
        /* cross correlation between data & x_est */
        for (i = 0 ; i < trans_sz ; i++){
            out[i][0] = data_Fconj[d][i][0]*F_est[i][0] - data_Fconj[d][i][1]*F_est[i][1] ;
            out[i][1] = data_Fconj[d][i][0]*F_est[i][1] + data_Fconj[d][i][1]*F_est[i][0] ;
        }
        fftw_execute(backward_plan) ;

        /* no or one particle */
        for (r = 0 ; r < num_step ; r++){
            idx = round(r*step_sz) ;
            L2 = data_sqsum[d] + model_sqsum[0][r] - in[idx]*f1 ;
            p_matrix[d][0][r] = log_prior[0][r] - L2*f2 ; 
        }

        /* two particles */
        for (r = 0 ; r < sep_ct ; r++){
            i0 = round(shift_table[r][0]*step_sz) ;
            j0 = round(shift_table[r][1]*step_sz) ;
            L2 = data_sqsum[d] + model_sqsum[1][r] - (in[i0] + in[j0])*f1 ;
            p_matrix[d][1][r] = log_prior[1][r] - L2*f2 ;
        }

        max_log_p = p_matrix[d][0][0] ;
        for (r = 1 ; r < num_step ; r++){
            if (max_log_p < p_matrix[d][0][r])
                max_log_p = p_matrix[d][0][r] ;
        }

        for (r = 0 ; r < sep_ct ; r++){
            if (max_log_p < p_matrix[d][1][r])
                max_log_p = p_matrix[d][1][r] ;
        }

        psum = 0. ;
        for (r = 0 ; r < num_step ; r++){
            p_matrix[d][0][r] -= max_log_p ;
            p_matrix[d][0][r] = exp(p_matrix[d][0][r]) ;
            psum += p_matrix[d][0][r] ;
        }

        for (r = 0 ; r < sep_ct ; r++){
            p_matrix[d][1][r] -= max_log_p ;
            p_matrix[d][1][r] = exp(p_matrix[d][1][r]) ;
            psum += p_matrix[d][1][r] ;
        }

        inv_psum = 1./psum ;
        for (r = 0 ; r < num_step ; r++)
            p_matrix[d][0][r] *= inv_psum ;
        for (r = 0 ; r < sep_ct ; r++)
            p_matrix[d][1][r] *= inv_psum ;
    }

    /* M-step */
    memset(x_est, 0, num_byte) ;
    memset(inter_weight, 0, num_byte) ;

    for (d = dmin ; d < dmax ; d++){
        
        memset(in, 0, num_byte) ;
        for (r = 0 ; r < num_step ; r++){
            idx = round(r*step_sz) ;
            in[idx] = p_matrix[d][0][r] ;
        }

        for (r = 0 ; r < sep_ct ; r++){
            i0 = round(shift_table[r][0]*step_sz) ;
            j0 = round(shift_table[r][1]*step_sz) ;
            in[i0] += p_matrix[d][1][r] ;
            in[j0] += p_matrix[d][1][r] ;
        }
        fftw_execute(forward_plan) ;

        for (i = 0 ; i < trans_sz ; i++){
            F_est[i][0] = out[i][0] ;
            F_est[i][1] = out[i][1] ;
            out[i][0] = F_est[i][0]*data_Fconj[d][i][0] + F_est[i][1]*data_Fconj[d][i][1] ;
            out[i][1] = F_est[i][1]*data_Fconj[d][i][0] - F_est[i][0]*data_Fconj[d][i][1] ;
        }

        fftw_execute(backward_plan) ;
        for (i = L ; i < trans_sz ; i++)
            x_est[i] += in[i] ;

        for (i = 0 ; i < trans_sz ; i++){
            out[i][0] = F_est[i][0]*unity_F[i][0] - F_est[i][1]*unity_F[i][1] ;
            out[i][1] = F_est[i][0]*unity_F[i][1] + F_est[i][1]*unity_F[i][0] ;
        }

        fftw_execute(backward_plan) ;
        for (i = L ; i < trans_sz ; i++)
            inter_weight[i] += in[i] ;
    }

    MPI_Allreduce(MPI_IN_PLACE, x_est, trans_sz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
    MPI_Allreduce(MPI_IN_PLACE, inter_weight, trans_sz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
    
    for (i = L ; i < trans_sz ; i++){
        if (inter_weight[i] > 0)
            x_est[i] /= inter_weight[i] ;
    }

    /* update psf */
    memset(psum_vec[0], 0, num_step*sizeof(double)) ;
    memset(psum_vec[1], 0, sep_ct*sizeof(double)) ;
    for (d = dmin ; d < dmax ; d++){
        for (r = 0 ; r < num_step ; r++)
            psum_vec[0][r] += p_matrix[d][0][r] ;
        for (r = 0 ; r < sep_ct ; r++)
            psum_vec[1][r] += p_matrix[d][1][r] ;   
    }

    for (r = 0 ; r < num_step ; r++)
        psum_vec[0][r] /= num_data ;
    for (r = 0 ; r < sep_ct ; r++)
        psum_vec[1][r] /= num_data ;

    MPI_Allreduce(MPI_IN_PLACE, &psum_vec[0][0], num_step, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
    MPI_Allreduce(MPI_IN_PLACE, &psum_vec[1][0], sep_ct, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;

    if (myid == root)
        update_psf( num_step ) ;
    MPI_Bcast(&psf[0], num_step/2 + 1, MPI_DOUBLE, root, MPI_COMM_WORLD) ;

    prior[0][0] = psf[num_step/2 - 1] ;
    prior[0][1] = psf[num_step/2] ;
    prior[0][num_step - 1] = psf[num_step/2] ;
    for (r = 2 ; r < num_step/2 + 1 ; r++){
        prior[0][r] = prior[0][r-1] + psf[num_step/2 - r] ;
        prior[0][num_step - r] = prior[0][r] ;
    }
    
    for (i = 0 ; i < sep_ct ; i++){
        r = shift_table[i][0] ;
        s = shift_table[i][1] ;
        idx = s - r - num_step/2 ;
        prior[1][i] = psf[idx] ;
    }

    if (myid == root){
        for (r = 0 ; r < num_step/2 + 1 ; r++)
            printf("%.7f ", prior[0][r]) ;
        printf("\n") ;

        for (r = 0 ; r < num_step/2 + 1 ; r++)
            printf("%.7f ", psf[r]) ;

        L2 = 0. ;
        for (r = 0 ; r < num_step/2 + 1 ; r++)
            L2 += pow(prev_psf[r], 2) ;
        if (L2 > 0.){
            err = 0. ;
            for (r = 0 ; r < num_step/2 + 1 ; r++)
                err += pow(psf[r] - prev_psf[r], 2) ;
            err = sqrt(err/L2) ;
        }
        else
            err = -1 ;
        printf("\t%1.6e", err) ;
    }

    err = 0. ;
    L2 = 0. ;
    for (i = L ; i < trans_sz ; i++){
        err += pow(x_est[i] - prev_x_est[i], 2) ;
        L2 += pow(prev_x_est[i], 2) ;
    }
    err = sqrt(err/L2) ;
    if (myid == root)
        printf("\t%1.6e\n", err) ;

    memcpy(prev_psf, psf, (num_step/2 + 1)*sizeof(double)) ;
    memcpy(prev_x_est, x_est, num_byte) ;
}


void update_psf( int num_step ){

    int i, r, s, idx, par_sz, sep_ct, idx_min ;
    int iter_ct, search_ct, max_iter = 1000 ;
    double x, dx, x0, x1, x2, x3, f0, f1, f2, f3, fmin ;
    double inv_tau, diff, norm, alpha, delta ;
    double tol = 1.e-7, epsilon = 1.e-10 ;
    
    par_sz = num_step/2 ;
    sep_ct = par_sz*(par_sz - 1)/2 ;
    inv_tau = 1/golden_ratio ;

    /* initialize at a feasible point */
    memset(&psf[0], 0, (par_sz+1)*sizeof(double)) ;
    for (i = 0 ; i < sep_ct ; i++){
        r = shift_table[i][0] ;
        s = shift_table[i][1] ;
        idx = s - r - par_sz ;
        psf[idx] += psum_vec[1][i] ;
    }
    
    for (r = 0 ; r < par_sz - 1 ; r++)
        psf[r] /= (par_sz - 1 - r) ;
    psf[par_sz - 1] = psum_vec[0][0] ;

    psf[par_sz] = 1 - psf[par_sz - 1] ;
    for (r = 0 ; r < par_sz - 1 ; r++)
        psf[par_sz] -= psf[r]*(par_sz + r) ;
    psf[par_sz] /= (2*par_sz - 1) ;

    if (psf[par_sz] < epsilon)
        printf("warning: psf[par_sz] <= 0 !!\n") ;

    /* optimize psf using the Frank-Wolfe algorithm */
    delta = 1. ;
    iter_ct = 0 ;
    while (delta > tol && iter_ct < max_iter){
        
        /* calculate gradient at the feasible point */
        prior[0][0] = psf[par_sz - 1] ;
        prior[0][1] = psf[par_sz] ;
        prior[0][num_step - 1] = psf[par_sz] ;
        for (r = 2 ; r < par_sz + 1 ; r++){
            prior[0][r] = prior[0][r-1] + psf[par_sz - r] ;
            prior[0][num_step - r] = prior[0][r] ;
        }
    
        for (i = 0 ; i < sep_ct ; i++){
            r = shift_table[i][0] ;
            s = shift_table[i][1] ;
            idx = s - r - par_sz ;
            prior[1][i] = psf[idx] ;
        }

        memset(&grad[0], 0, (par_sz+1)*sizeof(double)) ;
        for (s = 0 ; s < par_sz - 1 ; s++){
            for (r = par_sz - s ; r < par_sz + s + 1 ; r++)
                grad[s] -= psum_vec[0][r]/prior[0][r] ;
        }
        
        for (i = 0 ; i < sep_ct ; i++){
            r = shift_table[i][0] ;
            s = shift_table[i][1] ;
            idx = s - r - par_sz ;
            grad[idx] -= psum_vec[1][i]/prior[1][i] ;
        }
    
        grad[par_sz-1] = -psum_vec[0][0]/prior[0][0] ;
        for (r = 1 ; r < num_step ; r++)
            grad[par_sz] -= psum_vec[0][r]/prior[0][r] ;

        /* determine the search direction by minimizing the dot 
         * product of grad and the vertices of the convex set */

        idx_min = par_sz ;
        fmin = grad[par_sz]/(2*par_sz - 1) ;
        
        x = grad[par_sz-1]*1. ;
        if (x < fmin){
            fmin = x ;
            idx_min = par_sz - 1 ;
        }

        for (s = 0 ; s < par_sz - 1 ; s++){
            x = grad[s]/(par_sz + s) ;
            if (x < fmin){
                fmin = x ;
                idx_min = s ;
            }
        }
        
        for (s = 0 ; s < par_sz + 1 ; s++)
            search_vec[s] = -psf[s] ;

        if (idx_min == par_sz)
            search_vec[idx_min] += 1./(2*par_sz - 1) ;
        else if (idx_min == par_sz - 1)
            search_vec[idx_min] += 1. ;
        else
            search_vec[idx_min] += 1./(idx_min + par_sz) ;

        /* determine step size by golden section search */
        x0 = 0. ;
        x1 = 1. - 1.e-15 ;
        dx = x1 - x0 ;

        f0 = func_val(par_sz, x0) ;
        f1 = func_val(par_sz, x1) ;
        while (f1 > f0){
            x1 *= 0.1 ;
            f1 = func_val(par_sz, x1) ;
        }
        x1 *= 10. ;

        diff = 0. ;
        norm = 0. ;
        for (r = 0 ; r < par_sz - 1 ; r++){
            diff += search_vec[r]*search_vec[r] ;
            norm += psf[r]*psf[r] ;
        }
        diff = sqrt(diff) ;
        norm = sqrt(norm) ;

        search_ct = 0 ;
        while (diff*dx > norm*tol && search_ct < max_iter){
            dx = (x1 - x0)*inv_tau ;
            x2 = x1 - dx ;
            x3 = x0 + dx ;
            f2 = func_val(par_sz, x2) ;
            f3 = func_val(par_sz, x3) ;
            if (f2 < f3)
                x1 = x3 ;
            else
                x0 = x2 ;
            dx = x1 - x0 ;
            search_ct += 1 ;
        }
        
        /* update psf */
        delta = 0. ;
        norm = 0. ;
        alpha = (x0 + x1)*0.5 ;
        for (s = 0 ; s < par_sz + 1 ; s++){
            diff = alpha*search_vec[s] ;
            delta += diff*diff ;
            norm += psf[s]*psf[s] ;
            psf[s] += diff ;
        }

        delta = sqrt(delta/norm) ;
        iter_ct += 1 ;
    }
    printf("iter_ct = %d, delta = %1.10e\n", iter_ct, delta) ;
}


double func_val( int par_sz, double dx ){
    
    int i, r, s, idx, num_step, sep_ct ;
    double val, x ;

    num_step = 2*par_sz ;
    sep_ct = par_sz*(par_sz-1)/2 ;

    x = psf[par_sz - 1] + dx*search_vec[par_sz - 1] ;
    val = -psum_vec[0][0]*log(x) ;

    x = psf[par_sz] + dx*search_vec[par_sz] ;
    val -= (psum_vec[0][1] + psum_vec[0][num_step - 1])*log(x) ;
    for (r = 2 ; r < par_sz ; r++){
        x += psf[par_sz - r] + dx*search_vec[par_sz - r] ;
        val -= (psum_vec[0][r] + psum_vec[0][num_step - r])*log(x) ;
    }

    x += psf[0] + dx*search_vec[0] ;
    val -= psum_vec[0][par_sz]*log(x) ;

    for (i = 0 ; i < sep_ct ; i++){
        r = shift_table[i][0] ;
        s = shift_table[i][1] ;
        idx = s - r - par_sz ;
        val -= psum_vec[1][i]*log(psf[idx] + dx*search_vec[idx]) ;
    }

    return val ;
}


void redist_prior( int par_sz ){

    int i, j, r, s, new_par_sz, sep_ct ;
    double xi, yi, A, B, C, D, inv_det ;
    double density, step_sz, new_step_sz, rescale ;
    double *cpy_psf, *cpy_prior_0 ;
    double mat_A[2][2], inv_mat_A[2][2], vec_b[2], pmtr[2] ;

    density = prior[0][par_sz]*par_sz ;
    new_par_sz = par_sz + 2 ;
    if (new_par_sz > L)
        new_par_sz = L ;

    step_sz = ((double) L) / par_sz ;
    new_step_sz = ((double) L) / new_par_sz ;
    rescale = new_step_sz / step_sz ;

    mat_A[0][0] = 0. ;
    mat_A[0][1] = 0. ;
    mat_A[1][0] = 0. ;
    mat_A[1][1] = 0. ;
    vec_b[0] = 0. ;
    vec_b[1] = 0. ;

    /* fit y = pmtr[0]*x + pmtr[1] */
    for (i = 0 ; i < par_sz ; i++){
        xi = i*step_sz ;
        yi = prior[0][par_sz+i] ;
        mat_A[0][0] += xi*xi ;
        mat_A[0][1] += xi ;
        mat_A[1][0] += xi ;
        mat_A[1][1] += 1. ;
        vec_b[0] += xi*yi ;
        vec_b[1] += yi ;
    }

    A = mat_A[0][0] ;
    B = mat_A[0][1] ;
    C = mat_A[1][0] ;
    D = mat_A[1][1] ;
    inv_det = 1./(A*D - B*C) ;

    inv_mat_A[0][0] = D*inv_det ;
    inv_mat_A[0][1] = -B*inv_det ;
    inv_mat_A[1][0] = -C*inv_det ;
    inv_mat_A[1][1] = A*inv_det ;

    for (i = 0 ; i < 2 ; i++){
        pmtr[i] = 0. ;
        for (j = 0 ; j < 2 ; j++)
            pmtr[i] += inv_mat_A[i][j]*vec_b[j] ;
    }

    cpy_prior_0 = calloc(new_par_sz, sizeof(double)) ;
    for (i = 0 ; i < new_par_sz ; i++){
        xi = i*new_step_sz ;
        cpy_prior_0[i] = (pmtr[0]*xi + pmtr[1])*rescale ;
    }

    cpy_psf = calloc(new_par_sz+1, sizeof(double)) ;
    for (i = 0 ; i < new_par_sz - 1 ; i++)
        cpy_psf[i] = cpy_prior_0[i] - cpy_prior_0[i+1] ;
    cpy_psf[new_par_sz] = cpy_prior_0[new_par_sz-1] ;

    cpy_psf[new_par_sz-1] = 1 ;
    for (i = 0 ; i < new_par_sz - 1 ; i++)
        cpy_psf[new_par_sz-1] -= (i+new_par_sz)*cpy_psf[i] ;
    cpy_psf[new_par_sz-1] -= (2*new_par_sz-1)*cpy_psf[new_par_sz] ;

    for (i = 0 ; i < new_par_sz + 1 ; i++)
        psf[i] = cpy_psf[i] ;

    prior[0][0] = psf[new_par_sz - 1] ;
    prior[0][1] = psf[new_par_sz] ;
    prior[0][2*new_par_sz - 1] = psf[new_par_sz] ;
    for (i = 2 ; i < new_par_sz + 1 ; i++){
        prior[0][i] = prior[0][i-1] + psf[new_par_sz - i] ;
        prior[0][2*new_par_sz - i] = prior[0][i] ;
    }

    sep_ct = 0 ;
    for (r = 1 ; r < 2*new_par_sz ; r++){
        for (s = r+1 ; s < 2*new_par_sz ; s++){
            if (s-r < new_par_sz)
                continue ;
            shift_table[sep_ct][0] = r ;
            shift_table[sep_ct][1] = s ;
            sep_ct += 1 ;
        }
    }

    for (i = 0 ; i < sep_ct ; i++){
        r = shift_table[i][0] ;
        s = shift_table[i][1] ;
        prior[1][i] = psf[s - r - new_par_sz] ;
    }

    free(cpy_prior_0) ;
    free(cpy_psf) ;
}


void cal_likelihood( int num_step ){

    int i, d, m, r, s, idx, idx_start, num_byte ;
    int i0, j0, i1, j1, dmin, dmax, sep_ct ;
    double step_sz, L2, max_log_p ;
    double psum, inv_psum, f1, f2 ;

    num_byte = trans_sz*sizeof(double) ;
    memcpy(in, x_est, num_byte) ;
    step_sz = trans_sz / ((double) num_step) ;

    fftw_execute(forward_plan) ;
    for (i = 0 ; i < trans_sz ; i++){
        F_est[i][0] = out[i][0] ;
        F_est[i][1] = out[i][1] ;
    }

    /* no or one particle */
    for (r = 0 ; r < num_step ; r++){
        model_sqsum[0][r] = 0. ;
        idx_start = round(r*step_sz) ;
        for (i = 0 ; i < L ; i++){
            idx = (idx_start + i) % trans_sz ;
            model_sqsum[0][r] += x_est[idx]*x_est[idx] ;
        }
    }

    /* two particles */
    sep_ct = (num_step/2)*(num_step/2 - 1)/2 ;
    for (r = 0 ; r < sep_ct ; r++){
        model_sqsum[1][r] = 0. ;
        i0 = round(shift_table[r][0]*step_sz) ;
        j0 = round(shift_table[r][1]*step_sz) ;
        for (s = 0 ; s < L ; s++){
            i1 = (i0 + s) % trans_sz ;
            j1 = (j0 + s) % trans_sz ;
            model_sqsum[1][r] += pow(x_est[i1] + x_est[j1], 2) ;
        }
    }

    dmin = 0 ;
    dmax = s_num_data ;
    if (myid == nproc-1)
        dmax = num_data - (nproc-1)*s_num_data ;

    LL = 0. ;
    f1 = 2*inv_trans_sz ;
    f2 = 0.5*inv_var ;

    for (d = dmin ; d < dmax ; d++){
        /* cross correlation between data & x_est */
        for (i = 0 ; i < trans_sz ; i++){
            out[i][0] = data_Fconj[d][i][0]*F_est[i][0] - data_Fconj[d][i][1]*F_est[i][1] ;
            out[i][1] = data_Fconj[d][i][0]*F_est[i][1] + data_Fconj[d][i][1]*F_est[i][0] ;
        }
        fftw_execute(backward_plan) ;

        /* no or one particle */
        for (r = 0 ; r < num_step ; r++){
            idx = round(r*step_sz) ;
            L2 = data_sqsum[d] + model_sqsum[0][r] - in[idx]*f1 ;
            p_matrix[d][0][r] = -L2*f2 ;
        }

        /* two particles */
        for (r = 0 ; r < sep_ct ; r++){
            i0 = round(shift_table[r][0]*step_sz) ;
            j0 = round(shift_table[r][1]*step_sz) ;
            L2 = data_sqsum[d] + model_sqsum[1][r] - (in[i0] + in[j0])*f1 ;
            p_matrix[d][1][r] = -L2*f2 ;
        }

        max_log_p = p_matrix[d][0][0] ;
        for (r = 1 ; r < num_step ; r++){
            if (max_log_p < p_matrix[d][0][r])
                max_log_p = p_matrix[d][0][r] ;
        }

        for (r = 0 ; r < sep_ct ; r++){
            if (max_log_p < p_matrix[d][1][r])
                max_log_p = p_matrix[d][1][r] ;
        }

        psum = 0. ;
        for (r = 0 ; r < num_step ; r++){
            p_matrix[d][0][r] -= max_log_p ;
            p_matrix[d][0][r] = exp(p_matrix[d][0][r]) ;
            psum += p_matrix[d][0][r] ;
        }

        for (r = 0 ; r < sep_ct ; r++){
            p_matrix[d][1][r] -= max_log_p ;
            p_matrix[d][1][r] = exp(p_matrix[d][1][r]) ;
            psum += p_matrix[d][1][r] ;
        }

        inv_psum = 1./psum ;
        for (r = 0 ; r < num_step ; r++)
            p_matrix[d][0][r] *= inv_psum ;
        for (r = 0 ; r < sep_ct ; r++)
            p_matrix[d][1][r] *= inv_psum ;

        psum = 0. ;
        for (r = 0 ; r < num_step ; r++)
            psum += p_matrix[d][0][r]*prior[0][r] ;
        for (r = 0 ; r < sep_ct ; r++)
            psum += p_matrix[d][1][r]*prior[1][r] ;
        LL += log(psum) ;
    }

    MPI_Allreduce(MPI_IN_PLACE, &LL, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
}


void setup(){

    FILE *fp ; 
    int i, d, dmin, dmax, sep_ct ;
    double rescale, data_sum ;
    long long byte_offset ;
    time_t t0 ;

    fp = fopen(datafile, "rb") ;
    fread(&Npix, sizeof(long long), 1, fp) ;
    fread(&L, sizeof(int), 1, fp) ;
    fread(&sigma, sizeof(double), 1, fp) ;
    
    inv_var = 1./(sigma*sigma) ;
    trans_sz = 2*L ;
    inv_trans_sz = 1./trans_sz ;
    sep_ct = L*(L-1)/2 ;

    num_data = Npix/L ;
    if (num_data % nproc == 0)
        s_num_data = num_data / nproc ;
    else
        s_num_data = num_data / nproc + 1 ;

    byte_offset = myid*s_num_data ;
    byte_offset *= L*sizeof(double) ;
    fseek(fp, byte_offset, SEEK_CUR) ;

    dmin = 0 ;
    dmax = s_num_data ;
    if (myid == nproc-1)
        dmax = num_data - (nproc-1)*s_num_data ;

    data_sum = 0. ;
    data = malloc(s_num_data * sizeof(double *)) ;
    for (d = dmin ; d < dmax ; d++){
        data[d] = malloc(L * sizeof(double)) ;
        fread(data[d], sizeof(double), L, fp) ;
        for (i = 0 ; i < L ; i++)
            data_sum += data[d][i] ;
    }
    fclose(fp) ;

    if (myid == root)
        printf("L = %d, num_data = %lld, sigma = %lf\n", L, num_data, sigma) ;
    MPI_Allreduce(MPI_IN_PLACE, &data_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
    
    /* pair seperation function */
    psf = calloc(L+1, sizeof(double)) ;
    prev_psf = calloc(L+1, sizeof(double)) ;
    grad = malloc((L+1) * sizeof(double)) ;
    search_vec = malloc((L+1) * sizeof(double)) ;

    prior = malloc(num_class * sizeof(double *)) ;
    prior[0] = malloc(trans_sz * sizeof(double)) ;
    prior[1] = malloc(sep_ct * sizeof(double)) ;
    log_prior = malloc(num_class * sizeof(double *)) ;
    log_prior[0] = malloc(trans_sz * sizeof(double)) ;
    log_prior[1] = malloc(sep_ct * sizeof(double)) ;

    x_est = calloc(trans_sz, sizeof(double)) ;
    prev_x_est = calloc(trans_sz, sizeof(double)) ;
    inter_weight = malloc(trans_sz * sizeof(double)) ;

    /* initialize signal estimate */
    rescale = 0. ;
    srand( (unsigned) time(&t0) ) ;
    for (i = L ; i < trans_sz ; i++){
        x_est[i] = rand() ;
        rescale += x_est[i] ;
    }
    rescale = data_sum/(num_data*rescale) ;
    for (i = L ; i < trans_sz ; i++)
        x_est[i] *= rescale ;
    MPI_Bcast(x_est, trans_sz, MPI_DOUBLE, root, MPI_COMM_WORLD) ;

    p_matrix = malloc(s_num_data * sizeof(double **)) ;
    for (d = dmin ; d < dmax ; d++){
        p_matrix[d] = malloc(num_class * sizeof(double *)) ;
        p_matrix[d][0] = malloc(trans_sz * sizeof(double)) ;
        p_matrix[d][1] = malloc(sep_ct * sizeof(double)) ;
    }

    psum_vec = malloc(num_class * sizeof(double *)) ;
    psum_vec[0] = malloc(trans_sz * sizeof(double)) ;
    psum_vec[1] = malloc(sep_ct * sizeof(double)) ;

    in = calloc(trans_sz, sizeof(double)) ;
    out = fftw_malloc(trans_sz * sizeof(fftw_complex)) ;
    F_est = fftw_malloc(trans_sz * sizeof(fftw_complex)) ;
    unity_F = fftw_malloc(trans_sz * sizeof(fftw_complex)) ;
    data_Fconj = fftw_malloc(s_num_data * sizeof(fftw_complex *)) ;
    forward_plan = fftw_plan_dft_r2c_1d(trans_sz, in, out, FFTW_MEASURE) ;

    /* data_Fconj stores the conjugate of the data Fourier transform */
    for (d = dmin ; d < dmax ; d++){
        data_Fconj[d] = fftw_malloc(trans_sz * sizeof(fftw_complex)) ;
        memcpy(&in[0], data[d], L*sizeof(double)) ;
        fftw_execute(forward_plan) ;
        for (i = 0 ; i < trans_sz ; i++){
            data_Fconj[d][i][0] = out[i][0] ;
            data_Fconj[d][i][1] = -out[i][1] ;
        }
    }

    for (i = 0 ; i < L ; i++)
        in[i] = 1. ;
    fftw_execute(forward_plan) ;
    for (i = 0 ; i < trans_sz ; i++){
        unity_F[i][0] = out[i][0] ;
        unity_F[i][1] = out[i][1] ;
    }
 
    fftw_destroy_plan(forward_plan) ;
 
    model_sqsum = malloc(num_class * sizeof(double *)) ;
    model_sqsum[0] = malloc(trans_sz * sizeof(double)) ;
    model_sqsum[1] = malloc(sep_ct * sizeof(double)) ;
    shift_table = malloc(sep_ct * sizeof(*shift_table)) ;

    data_sqsum = calloc(s_num_data, sizeof(double)) ;
    for (d = dmin ; d < dmax ; d++){
        for (i = 0 ; i < L ; i++)
            data_sqsum[d] += data[d][i]*data[d][i] ;
    }
}


void free_mem(){

    int d, k, dmin, dmax ;
    dmin = 0 ;
    dmax = s_num_data ;
    if (myid == nproc-1)
        dmax = num_data - (nproc-1)*s_num_data ;

    free(psf) ;
    free(prev_psf) ;
    free(grad) ;
    free(search_vec) ;
    free(x_est) ;
    free(prev_x_est) ;
    free(inter_weight) ;

    for (d = dmin ; d < dmax ; d++){
        free(data[d]) ;
        for (k = 0 ; k < num_class ; k++)
            free(p_matrix[d][k]) ;
        free(p_matrix[d]) ;
        fftw_free(data_Fconj[d]) ;
    }
    free(data) ;
    free(p_matrix) ;
    fftw_free(data_Fconj) ;

    for (k = 0 ; k < num_class ; k++){
        free(prior[k]) ;
        free(log_prior[k]) ;
        free(psum_vec[k]) ;
        free(model_sqsum[k]) ;
    }
    free(prior) ;
    free(log_prior) ;
    free(psum_vec) ;
    free(model_sqsum) ;
    free(shift_table) ;

    free(in) ;
    free(data_sqsum) ;
    fftw_free(out) ;
    fftw_free(F_est) ;
    fftw_free(unity_F) ;
}
