function load_dict1_u(datafile; num_type=Int)
    # load *.tsv
    # Dict[(k1) => Int or Float64]
 #   if num_type == Int
        dict1 = Dict{String, Int64}()
        #func(count::String) = parse(Int,count)
 #   else
 #       dict1 = Dict{String, Float64}()
 #       func(count::SubString) = parse(Float64,count)
 #   end
    fid = open( datafile )
    # lines_src = countlines(datafile); chunk = 1_000_000
    while !eof(fid)
        line = readline(fid)
        #line = strip(readline(fid))
        #line=strip(line)
        try
          #if !isempty(line) 
            (k1, count) = split(chomp(line),'\t')
            #dict1[k1] = func(count)
            dict1[k1] = parse(Int64,count)
          #else #println(line)    
        #end
      catch 
      end    
    end
    close(fid)
    return dict1 
end