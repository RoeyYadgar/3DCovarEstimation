function tensor = transposeTensor(tensor,dim1,dim2)

    if(~exist('dim1'))
        dim1 = 1;
    end
    if(~exist('dim2'))
        dim2 = 2;
    end

    dims = 1:ndims(tensor);
    dims(dim1) = dim2;
    dims(dim2) = dim1;

    tensor = permute(tensor,dims);



end