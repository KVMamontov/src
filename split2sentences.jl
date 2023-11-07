function split2sentences(input_file; type="sentences")
    # Golem2.jl text=""
    # phrases = split(text, r"(?<=[^\s\.]{2}[.,;:!\?])[\s\\]")
    # phrases = split(text, r"(?<=[^\s\.]{2}[.!\?])[\s\\]")
    # (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s
    # Aug2023 - TODO ."  !"  ?"  ..." 
    # split2sentences("webtext2_1M.txt")
    t0=time()
    output_file = split(input_file, ".")[1] * "_sentences.txt"
    lines_in = 0; lines_out = 0 
    if type == "sentences"
        #regex = r"(?<=[^\s\.]{2}[.!\?])[\s\\]" # \s -> space ?<=
        #regex = r"(?<=[.!?])\s+(?=[A-Z])"
        regex = r"(?<=[^\s\.]{2}[.!\?])[!\"\?\"\.\")[\s\\]"
    elseif type == "phrases"
        regex = r"(?<=[^\s\.]{2}[.,;:!\?])[!\"][\s\\]"
    end  
    io = IOBuffer();   #io2 = IOBuffer()
    fid = open(input_file); chunk = 1_000_000
    lines_src = countlines(input_file)
    while !eof(fid)
        line = readline(fid) # if keep=true -> dont remove trailing \n or \r\n
        lines_in +=1;
        #s = split(line, r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
        #s = split(line, r"(?<=[^\s\.]{2}[.!\?])[\s\\]")
        s = split(line, regex)
        newline = ""
        if length(s)>0
            for i in 1:length(s)
                #print(io, s[i] * '\n'); lines_out += 1 # restore removed trailing \n
                newline = s[i] ; lines_out += 1 # restore removed trailing \n
                newline = lstrip(newline)
                newline = rstrip(newline)
                newline = lowercase(newline)
                newline = uppercasefirst(newline)
                print(io, newline * '\n'); lines_out += 1 # restore removed trailing \n
            end  
            #else print(io2, line * '\n')  
        end
        
        if lines_in%chunk == 0
            t_ = chunk * ((time() - t0) / lines_in)
            t_ = round(t_, digits=3)
            done = round(100*(lines_in / lines_src), digits=2)
            println(done,"%  lines_out=", lines_out, "  t_chunk=", t_)
            flush(stdout)
        end
    end
    text = String(take!(io))
    write(output_file, text) # "input_filename._tmp"
    #removed = words_in - words_out
    #println("lines in=", lines_in, "    lines out=", lines_out);
    #println("total words=", words_in, "    removed=", removed,  "  ",(removed/words_in)*100, " %")
    println(output_file, " saved")
    inbytes = filesize(input_file); outbytes = filesize(output_file)
    #println("input_file_size=", inbytes, "  output_file_size=", outbytes, " prop=", outbytes/inbytes)
    t1=time()
    #println("total time ",round((t1-t0)/3600,digits=3)," hours")
end