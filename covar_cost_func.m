function cost_val = covar_cost_func(y,Pu)
    
    L = size(Pu,1);
    r = size(Pu,3);
    n = size(Pu,4);
    y = reshape(y,L,L,1,n);

    norm_y_term = squeeze(sum(y.*y,[1,2])).^2;

    y_Pu_inner_prod = (sum(y.*Pu,[1,2]));
    y_Pu_term = squeeze(sum(y_Pu_inner_prod.^2,3));
    
    
    
    Pu = reshape(Pu,L^2,r,n);
    Pu_gram = pagemtimes(Pu,'transpose',Pu,'none');
    Pu_term = squeeze(sum((Pu_gram).^2,[1,2]));

    cost_val = mean(norm_y_term - 2*y_Pu_term + Pu_term)/(L^4); %validate L^4 normalization

end