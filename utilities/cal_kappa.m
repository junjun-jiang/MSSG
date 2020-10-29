function kappa = cal_kappa(conf_mat)

%kappa coefficient
PA = sum(diag(conf_mat));
n1 = sum(conf_mat,1);
n2 = sum(conf_mat,2);
PE = n1 * n2;
n = sum(conf_mat(:));
if (n*PA-PE) == 0 && (n^2-PE) == 0
    % Solve indetermination
    %warning('0 divided by 0')
    kappa = 1;
else
    kappa  = (n*PA-PE)/(n^2-PE);
end