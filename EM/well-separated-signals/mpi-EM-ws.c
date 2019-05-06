#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <mpi.h>

const int root = 0 ;
const int num_class = 1 ;
const double golden_ratio = 1.618034 ;
char datafile[256] = "micrograph.bin" ;

int nproc, myid, L, trans_sz ;
double *x_est, *prev_x_est ;
double *inter_weight, **data, **prior, **log_prior ;
double ***p_matrix, *in ;
double **model_sqsum, *data_sqsum ;
double sigma, inv_var, inv_trans_sz, err, LL ;
long long Npix, num_data, s_num_data ;
fftw_complex *out, *F_est, **data_Fconj, *unity_F ;
fftw_plan forward_plan, backward_plan ;

void setup() ;
void freq_marchingEM() ;
void exp_max( int ) ;
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
    int k, m, n, n_max, num_step, iter_ct ;
    int min_iter = 50, max_iter = 10000 ;
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

        /* initialize prior */
        if (n == 1){
            /* no or one particle */
            for (k = 0 ; k < num_step ; k++)
                prior[0][k] = 1./num_step ;
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
            printf("\nnum_step = %d: elapsed time = %.3f sec\n\n\n", num_step, t2-t1) ;
        }
    }
}


void exp_max( int num_step ){

    int i, d, r, idx, idx_start ;
    int dmin, dmax, num_byte ;
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

        max_log_p = p_matrix[d][0][0] ;
        for (r = 1 ; r < num_step ; r++){
            if (max_log_p < p_matrix[d][0][r])
                max_log_p = p_matrix[d][0][r] ;
        }

        psum = 0. ;
        for (r = 0 ; r < num_step ; r++){
            p_matrix[d][0][r] -= max_log_p ;
            p_matrix[d][0][r] = exp(p_matrix[d][0][r]) ;
            psum += p_matrix[d][0][r] ;
        }

        inv_psum = 1./psum ;
        for (r = 0 ; r < num_step ; r++)
            p_matrix[d][0][r] *= inv_psum ;
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

    /* update prior */    
    for (r = 0 ; r < num_step ; r++)
        prior[0][r] = 0. ;

    for (d = dmin ; d < dmax ; d++){
        for (r = 0 ; r < num_step ; r++)
            prior[0][r] += p_matrix[d][0][r] ;
    }

    for (r = 0 ; r < num_step ; r++)
        prior[0][r] /= num_data ;

    MPI_Allreduce(MPI_IN_PLACE, &prior[0][0], num_step, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;

    psum = 0. ;
    for (r = 1 ; r < num_step ; r++)
        psum += prior[0][r] ;
    
    psum /= num_step - 1 ;
    for (r = 1 ; r < num_step ; r++)
        prior[0][r] = psum ;

    if (myid == root){
        for (r = 0 ; r < num_step ; r++)
            printf("%.7f ", prior[0][r]) ;
        printf("\n") ;
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

    memcpy(prev_x_est, x_est, num_byte) ;
}


void redist_prior( int par_sz ){

    int r, new_par_sz, new_num_step ;
    double p0, step_sz, new_step_sz, rescale ;

    new_par_sz = par_sz + 2 ;
    if (new_par_sz > L)
        new_par_sz = L ;

    step_sz = ((double) L) / par_sz ;
    new_step_sz = ((double) L) / new_par_sz ;
    rescale = new_step_sz / step_sz ;
    new_num_step = 2*new_par_sz ;

    p0 = prior[0][0]*rescale ;
    for (r = 1 ; r < new_num_step ; r++)
        prior[0][r] = (1. - p0) / (new_num_step - 1) ;
}


void cal_likelihood( int num_step ){
    
    int i, d, r, idx, idx_start ;
    int dmin, dmax, num_byte ;
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

    for (r = 0 ; r < num_step ; r++){
        model_sqsum[0][r] = 0. ;
        idx_start = round(r*step_sz) ;
        for (i = 0 ; i < L ; i++){
            idx = (idx_start + i) % trans_sz ;
            model_sqsum[0][r] += x_est[idx]*x_est[idx] ;
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

        max_log_p = p_matrix[d][0][0] ;
        for (r = 1 ; r < num_step ; r++){
            if (max_log_p < p_matrix[d][0][r])
                max_log_p = p_matrix[d][0][r] ;
        }

        psum = 0. ;
        for (r = 0 ; r < num_step ; r++){
            p_matrix[d][0][r] -= max_log_p ;
            p_matrix[d][0][r] = exp(p_matrix[d][0][r]) ;
            psum += p_matrix[d][0][r] ;
        }

        inv_psum = 1./psum ;
        for (r = 0 ; r < num_step ; r++)
            p_matrix[d][0][r] *= inv_psum ;
        
        psum = 0. ;
        for (r = 0 ; r < num_step ; r++)
            psum += p_matrix[d][0][r]*prior[0][r] ;
        LL += log(psum) ;
    }

    MPI_Allreduce(MPI_IN_PLACE, &LL, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
}


void setup(){

    FILE *fp ; 
    int i, d, dmin, dmax ;
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
    
    prior = malloc(num_class * sizeof(double *)) ;
    prior[0] = malloc(trans_sz * sizeof(double)) ;
    log_prior = malloc(num_class * sizeof(double *)) ;
    log_prior[0] = malloc(trans_sz * sizeof(double)) ;

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
    }

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
        free(model_sqsum[k]) ;
    }
    free(prior) ;
    free(log_prior) ;
    free(model_sqsum) ;

    free(in) ;
    free(data_sqsum) ;
    fftw_free(out) ;
    fftw_free(F_est) ;
    fftw_free(unity_F) ;
}
