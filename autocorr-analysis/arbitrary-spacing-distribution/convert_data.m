sigma_list = [-1.5:0.1:0.5]' ;
length = size(sigma_list, 1) ;
num_run = 20 ;
num_trial = 10 ;

L = 10 ;
n_max = floor(0.5*L) ;
for m = 1:length
    outfile = sprintf('Data/repos/sigma_%.1f/output.dat', sigma_list(m)) ;
    fout = fopen(outfile, 'w') ;
    for r = 0:num_run-1
        for s = 0:num_trial-1
            infile = sprintf('Data/repos/sigma_%.1f/pamameters-%03d-0.mat', sigma_list(m), r) ;
            load(infile) ;
            for j = 1:n_max
                for i = 1:L
                    fprintf(fout, '%1.6e ', sol_x(i, j)) ;
                end
                fprintf(fout, '\n') ;
                for i = 1:L
                    fprintf(fout, '%1.6e ', sol_rho(i, j)) ;
                end
                fprintf(fout, '\n') ;
            end
            fprintf(fout, '%1.10e\n', stats(size(stats, 2)).cost) ;
        end
    end
    fclose(fout) ;
end
