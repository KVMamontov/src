# using LanguageFinder



# content
  # originally for 0.5.2; not all tested with 1.3.1
  # function written in 2023 (top-20) have been tested in 1.6.7
  # Golem_utils.jl utils for convert and prepare some input articles formats
  # using Plot; prepare PPL for plotting, read_dlm using DelimitedFiles
  # Naive Bayes on words and on sems; calc Mutual Information
  # create stopwords files from Sphinx -buildfreqs
  # convert csv  to WIKI xml-format
  # convert json to WIKI xml-format
  # ODBC to MySql, PostgreSQL
  # convert dat->tsv; function dat_tsv(datafile)
  # Packages used in 052: JSON, ODBC, CSV
  # Packages used in 131: DelimitedFiles, Plots 
  # Packages used in 167: TextAnalyses, Telegram, HTTP, StatsBase
  # function gz_csv_to_xml(input_file,output_file)
  # function gz_csv_content_to_xml(input_file,output_file)
  # function subtract_words(input_file,output_file,words_file)
  # function prepare_stops(input_words,output_words)
  # Adds stop-words starting in uppercase; if exists "the" adds "The" Sphinx
  # function build_form()
  # function sort_tsv(datafile)
  # function remove_comments(input_file,output_file)
  # function json_to_xml(input_file,output_file) Sputnik, Lenta
  # function clean_json(input_file,output_file) Sputnik



  #  using TextAnalysis, Languages
  #vocab_arr = readlines("wiki.vocab")
function ent()
  #fd = FileDocument("a_ru_10K.utf8"); language(fd)
  #fd = StringDocument("a_ru_10K.utf8")
  #fd = Document("a_ru_10K.utf8")
  #tx= text(fd)
  #ng=ngrams(fd,2,5)
  #ng = TextAnalysis.ngramize(Languages.Russian(), vocab_tokens,1,2,3) 
  #ng = TextAnalysis.onegramize(Languages.English(), vocab_tokens) 
  # words = ["1","2","3","2"] 
  #  TextAnalysis.frequencies(train)
  #append!(vocab_tokens, tokenize(Languages.English(),vocab_arr[i]))
  
  #  function read_()
  #  vocab_arr = readlines("en_27_sym.txt"); vocab_tokens = Vector{String}(undef,0)  
  #  for i in 1:length(vocab_arr)
  #    append!(vocab_tokens, tokenize(Languages.Russian(),vocab_arr[i]))
  #  end 
  
  #train_arr = readlines("wiki.train")
  train_arr = readlines("enwiki_10K_sym.txt"); train_tokens = Vector{String}(undef,0)
  for i in 1:length(train_arr)
    append!(train_tokens, tokenize(Languages.Russian(),train_arr[i]))
  end 
  score_arr = readlines("enwiki_100K_sym.txt"); score_tokens = Vector{String}(undef,0)
  for i in 1:length(score_arr)
    append!(score_tokens, tokenize(Languages.Russian(),score_arr[i]))
  end 
  return vocab_tokens, train_tokens, score_tokens
end



function tsv2ngrams(fid)
  # convert dict "abc" => count  to array ["a b c", "d e f", ... ]  
  # if count > 1 "a b c" repeats n times
  #fid = "d123_enwiki_100K.tsv"  fid =  "w123.tsv"
  # ngrams = tsv2ngrams(fid)
  d = load_dict1_u(fid); ngrams = Vector{String}(undef, 0) #"en_27_sym.tsv"
  for kk in keys(d)
    count = get(d, kk, 0)
    for v in 1:1 #count # 1 or count 
    #push!(ngrams, join(split(kk,""), " ") )  # split words into letters
    push!(ngrams, join(split(kk,""), "") )    # not split words
    end
  end  
  return ngrams
end



function txt2array(fid)
  # vocab_tokens, train_tokens, score_tokens = read_()
  lines = readlines(fid); train_tokens = Vector{String}(undef,0)
  for i in 1:length(lines)
    #append!(train_tokens, split(lines[i], "") ) #split into chars
    append!(train_tokens, split(lines[i], " ") ) #split into words
  end 
  println("words raw=", length(train_tokens))
  filter!(x->x!="",train_tokens)
  println("words no empties=", length(train_tokens))
  filter!(x->x!="<unk>",train_tokens) 
  println("words no <unk>=", length(train_tokens))
  return train_tokens
end


#=
  fid = "wiki.train.raw"  #"gz_markup.txt" #"enwiki_1K.txt" 
  train_tokens = txt2array(fid) 
  fid = "wiki.valid.raw"  #"gz_markup.txt" #"enwiki_1K.txt" 
  valid_tokens = txt2array(fid) 

  # train_tokens = sort(load_tsv2array("en_27_sym.tsv"))
  # train_ngrams = ngramizenew(sort(load_tsv2array("d1.tsv")),3)
  train_tokens = ["a","b","c","d","e"]
  train_tokens = ["a","b","c","c","b","d"]
  pop!(train_tokens)
  ng = ngramizenew(train_tokens,1)
  ng = ngramizenew(train_tokens,2)
  unique(ng)
  #train_tokens = tokenize(join(train_ngrams, " "));   #load_tsv2array("d1.tsv"))
  #ng = everygram(train_tokens,min_len=3,max_len=3)
    # vocab = Vocabulary(vocab_tokens, 1)
    # lookup(vocab,["===Roads===","Roads","roads"])  ;   vocab.vocab["the"]
  model = Laplace(train_tokens); #,1,"<unk>")
  fit = model(train_tokens,1,1);
  length(model.vocab.vocab)   # pop!(model.vocab.vocab)
    ppl = perplexity(model,fit, ngrams)
    #ppl = perplexity(model,fit, train_ngrams)
    # max PPL = model.vocab.vocab
    # max PPL = unique(model.vocab.vocab)
    # model.vocab.vocab["Robert"] == get(model.vocab.vocab,"Robert",0) 
    # fit.d["in free"]; "<unk>" in train_tokens; "= Robert Boulter" in ng

    ppl = perplexity(model,fit,["a"])
    ppl = perplexity(model,fit, train_tokens)
    ppl = perplexity(model,fit, ng)
    ppl = perplexity(model,fit, unique(ng))
    pop!(fit.d,"")
    ppl = perplexity(model,fit,["a b c","d e f"])
    ppl = perplexity(model,fit,["= Robert Boulter"])
    ppl = perplexity(model,fit,["области архитектуры и"])
    ppl = perplexity(model,fit,["c h i"])
    ppl = perplexity(model,fit,["t h e"])
    ppl = perplexity(model,fit,["i # @"])
        
    ppl = perplexity(model,fit,["the way he"])
    ppl = perplexity(model,fit,["= = ="])
    ppl = perplexity(model,fit,["= = The"])
    ppl = perplexity(model,fit,["= = In"])
    ppl = perplexity(model,fit,["Cy Young2 runner"])
    ppl = perplexity(model,fit,["of the <unk>"])
    ppl = perplexity(model,fit,[". = =",". = =",". = =",". = ="])
    ppl = perplexity(model,fit,[". = =",". = ="])
    ppl = perplexity(model,fit,["one of the"])
    ppl = perplexity(model,fit,["one of the","one of his"])
    ppl = perplexity(model,fit,["one of his"])
    ppl = perplexity(model,fit,["one of many"])
    ppl = perplexity(model,fit,["in free verse"])
    ppl = perplexity(model,fit,["it is as"])

    ppls =string( 1 / perplexity(model,fit,vocab_tokens))
    ppl_trains =string( 1 / perplexity(model,fit,train_tokens))
    write("ppl_result", ppls * "  " * ppl_trains) #, ppl_train)
    return ppl, ppl_train



    function entropy_test_string()
      #fit = model(train,1,1)  
      #mscr = maskedscore(model,fit,"севере","на")


      #=
             scr = score(model,fit,"e","t h")
             scr = score(model,fit,"i","t h")
             scr = score(model,fit,"y","t h")
             scr = score(model,fit,"y","t h")
             scr = score(model,fit,"h","п")
             scr = score(model,fit,"m","z q")
      mscr = maskedscore(model,fit,"i","t h")
      mscr = maskedscore(model,fit,"e","t h")
      lscr = logscore(model,fit,"i","t h")
      ent = entropy(model,fit,["t h y"])
      ent = entropy(model,fit,["t h e"])
      ent = entropy(model,fit,["s t a t"])
      ent = entropy(model,fit,["s t a q"])
      ent = entropy(model,fit,["t h e", "t h i"])
      ent = entropy(model,fit,["t h e", "t h o"])
      ent = entropy(model,fit,["t h i", "t h i"])
      ent = entropy(model,fit,["t h e", "t h e"])
      ent = entropy(model,fit,["t h e", "t h r"])
      ent = entropy(model,fit,["t h e", "t h i"])
      ent = entropy(model,fit,ng)
      
      scr = score(model,fit,"а","л")
      ent = entropy(model,fit,["а л"])
      ent = entropy(model,fit,["л а"])
      
      prb = score(model,fit,"р","п")
      ent = entropy(model,fit,["п р"])
      
      ent = entropy(model,fit,["с т"])
      ent = entropy(model,fit,["с"])
  
      scr = score(model,fit,"о","р")
      ent = entropy(model,fit,["р о"])
      
      scr = score(model,fit,"я","ш л") 
      ent = entropy(model,fit,["ш л я"])
      ent = entropy(model,fit,["ш л я","л я х"])
      ent = entropy(model,fit,["р о","я х"])
      ent = entropy(model,fit,["о и","р о","я х"])
      
      ent = entropy(model,fit,[" л х"])
      ent = entropy(model,fit,["л я"])
      ent = entropy(model,fit,["шля","лях"])
      ent = entropy(model,fit,["ш л","с х"])
      ent = entropy(model,fit,["ш л","с х","л а"])
      ent = entropy(model,fit,["ш л","с х","л а","<unk>","о"])
      
      ent = entropy(model,fit,train_tokens)
      ent = entropy(model,fit,vocab_tokens)
      =#


    end 
end
=#



function count_char_ngrams()
    # alphabet = #[Char.(Int('А'):Int('Е')); ['Ё']; Char.(Int('Ж'):Int('Я'));
    # [Char.(Int('а'):Int('е')); ['ё'];  Char.(Int('ж'):Int('я'));
    # [' '] ] 
    alphabet = [Char.(Int('A'):Int('Z'));
    Char.(Int('a'):Int('z')) ] #;  [' '] ] 
    
    #datafile="a_ru_100K.utf8"  enwiki_1K.txt webtext2_10M.txt
    io = open("enwiki_10K.txt","r")
    r = read(io,String);  s = collect(r);
    d1 = Dict(); d12 = Dict(); d123 = Dict(); 
    d1234 = Dict(); d12345 = Dict(); d123456 = Dict()
    for i in 1:length(s)-5 
      s1 = s[i]
      
      if Char(s1) in alphabet
        d1[ s1 ] = get(d1, s1, 0) + 1
        
        if Char(s[i+1]) in alphabet
          s12 = s[i] * s[i+1]
          d12[s12] = get(d12, s12, 0) + 1
          
          if Char(s[i+2]) in alphabet
            s123 = s12 * s[i+2]
            d123[s123] = get(d123, s123, 0) + 1 
          
          #=  
            if Char(s[i+3]) in alphabet
              s1234 = s123 * s[i+3]
              d1234[s1234] = get(d1234, s1234, 0) + 1
              
              if Char(s[i+4]) in alphabet
                s12345 = s1234 * s[i+4]
                d12345[s12345] = get(d12345, s12345, 0) + 1
                
                if Char(s[i+5]) in alphabet
                  s123456 = s12345 * s[i+5]
                  d123456[s123456] = get(d123456, s123456, 0) + 1
                end    
              end  
            end  
          =#
          
          end
        end    
      end
    end
    save_dict1_u(d1, "d1.tsv")
    save_dict1_u(d12, "d12.tsv")
    save_dict1_u_by2_desc(d123, "d123.tsv")  
    #save_dict1_u(d1234, "d1234.tsv") 
    #save_dict1_u(d12345, "d12345.tsv")  
    #save_dict1_u(d123456, "d123456.tsv")  
    # return d1 #, d12, #, d123, d3_desc
end  



function clean_log(input_file)
    # remove strings startswith "K-means" & startswith digit
    # input_file = "log_full_23_08_07_2"
    output_file = split(input_file, ".")[1] * "_clean"
    io_write = IOBuffer() 
    lines = readlines(input_file)
    for line in lines
      if length(line)>0 && !startswith(line, "K-means") && !isdigit(line[1])
          print(io_write, line * "\n")
      end  
    end  
    text = String(take!(io_write))
    write(output_file, text ); 
end  



function generation2dict(input_file)
    # корольков ||каллиадес ||шпунтик 
    # input_file = "1_gen.txt"
    output_file = split(input_file, ".")[1] * ".tsv"
    d_words = Dict("abc"=>1); empty!(d_words)
    io_write = IOBuffer()
    line = read(input_file, String)
    words = split(line, "||")
    for word in words
      word = split(word)[1] # remove leading and trailing spaces
      d_words[ word ] = get(d_words, word, 0) + 1       
    end
    save_dict1_u_by2_desc(d_words, output_file)
end



function unzip_src(input_file)
    src = input_file
    run(`ls -la`)
    run(`unzip`)
end



function make_word_ngrams(fid)
  # fid = "webtext2_10M.txt"
  words = String[]
  lines = readlines(fid)
    for i in 1:length(lines)
      new_words = split(lines[i])
      words = append!(words, new_words)
    end
    println("words=", length(words))
  
    dw1 = Dict()  # dw1 = Dict("word"=>1)
    for word in words
      word = string(word)
      dw1[word] = get(dw1, word, 0) + 1
    end

    total = sum(collect(values(dw1)))
    push!(dw1, "total 1-grams =" => total )
    push!(dw1, "distinct 1-grams =" => length(dw1) )
    dw1_desc = sort(collect(dw1), by=x->x[2], rev=true)
    println("dw1=", length(dw1))
    save_dict1_u_by2_desc(dw1_desc, "words1gram.tsv")
    dw1 = Dict(); GC.gc(); flush(stdout)


    dw2 = Dict()
    for i in 1:length(words) - 1
      word2 = string(words[i] * " " * words[i+1])
      dw2[word2] = get(dw2, word2, 0) + 1
    end
    
    total = sum(collect(values(dw2)))
    push!(dw2, "total 2-grams =" => total )
    push!(dw2, "distinct 2-grams =" => length(dw2) )
    dw2_desc = sort(collect(dw2), by=x->x[2], rev=true)
    println("dw2=", length(dw2))
    save_dict1_u_by2_desc(dw2_desc, "words2gram.tsv")
    dw2 = Dict(); GC.gc(); flush(stdout)

    dw3 = Dict()
    for i in 1:length(words) - 2
      word3 = string(words[i] * " " * words[i+1] * " " * words[i+2])
      dw3[word3] = get(dw3, word3, 0) + 1
    end

    total = sum(collect(values(dw3)))
    push!(dw3, "total 3-grams =" => total )
    push!(dw3, "distinct 3-grams =" => length(dw3) )
    dw3_desc = sort(collect(dw3), by=x->x[2], rev=true)
    println("dw3=", length(dw3))
    save_dict1_u_by2_desc(dw3_desc, "words3gram.tsv")
    dw3 = Dict(); GC.gc(); flush(stdout)

    dw4 = Dict()
    for i in 1:length(words) - 3
      word4 = string(words[i] * " " * words[i+1] * " " * words[i+2] * " " * words[i+3])
      dw4[word4] = get(dw4, word4, 0) + 1
    end
    
    total = sum(collect(values(dw4)))
    push!(dw4, "total 4-grams =" => total )
    push!(dw4, "distinct 4-grams =" => length(dw4) )
    dw4_desc = sort(collect(dw4), by=x->x[2], rev=true)
    println("dw4=", length(dw4))
    dw4 = Dict(); GC.gc(); flush(stdout)
    save_dict1_u_by2_desc(dw4_desc, "words4gram.tsv")

  #println("length(d_words)=",length(d_words2))
  #println("length(words)=",length(words))
  #d_sorted = sort(collect(d_words), by=x->x[2], rev=true)

  #return words , d_words_desc, d_words2_desc,  d_words3_desc,  d_words4_desc
  #return dw1_desc, dw2_desc, dw3_desc #, dw4_desc
end



function count_word_ngrams(input_file)
  # input_file ="webtext2_10M.txt"

  t0 = time()
  dw1 = Dict("word"=>1); empty!(dw1)
  dw2 = Dict("word"=>1); empty!(dw2)
  dw3 = Dict("word"=>1); empty!(dw3)
  dw4 = Dict("word"=>1); empty!(dw4)
  dw5 = Dict("word"=>1); empty!(dw5)
  dw6 = Dict("word"=>1); empty!(dw6)
  dw7 = Dict("word"=>1); empty!(dw7)
  
  # io = IOBuffer()
  fid = open(input_file); chunk = 10_000_000
  bytes_src = filesize(input_file) #  ~one-byte chars
  chars_in = 0
  words =["1","2","3","4","5","6","7"]
    
  while !eof(fid)  
      word = ""; eow = false;      
      while !eow && !eof(fid)
        sym = read(fid,Char); chars_in +=1;        
        if sym != ' ' && sym !='\r' && sym !='\n'
          word = word * sym
        else eow = true
          if word !="" #&& word !=''
          popfirst!(words)
          push!(words, word)
          n1 = words[1]
          dw1[n1] = get(dw1, n1, 0) + 1
          
          n2 = words[1] * ' ' * words[2]
          dw2[n2] = get(dw2, n2, 0) + 1
          
          n3 = words[1] * ' ' * words[2] * ' ' * words[3]
          dw3[n3] = get(dw3, n3, 0) + 1
          
          n4 = words[1] * ' ' * words[2] * ' ' * words[3] * ' ' * words[4]
          dw4[n4] = get(dw4, n4, 0) + 1
          
          n5 = words[1] * ' ' * words[2] * ' ' * words[3] * ' ' * words[4] * ' ' * words[5]
          dw5[n5] = get(dw5, n5, 0) + 1
          
          n6 = words[1] * ' ' * words[2] * ' ' * words[3] * ' ' * words[4] * ' ' * words[5]* ' ' * words[6]
          dw6[n6] = get(dw6, n6, 0) + 1
          
          n7 = words[1] * ' ' * words[2] * ' ' * words[3] * ' ' * words[4] * ' ' * words[5]* ' ' * words[6]* ' ' * words[7]
          dw7[n7] = get(dw7, n7, 0) + 1

          break
          end
        end  
      end  

      if chars_in%chunk == 0
        t_ = chunk * ((time() - t0) / chars_in)
        t_ = round(t_, digits=3)
        done = round(100*(chars_in / bytes_src), digits=2)
        println(done," %  chars_in=", chars_in, "  t_chunk=", t_)
        flush(stdout)
      end

  end  
  total = sum(collect(values(dw1)))
  push!(dw1, "total 1-grams =" => total )
  push!(dw1, "distinct 1-grams =" => length(dw1) )
  save_dict1_u_by2_desc(dw1, "words1gram.tsv")

  total = sum(collect(values(dw2)))
  push!(dw2, "total 2-grams =" => total )
  push!(dw2, "distinct 2-grams =" => length(dw2) )
  save_dict1_u_by2_desc(dw2, "words2gram.tsv")

  total = sum(collect(values(dw3)))
  push!(dw3, "total 3-grams =" => total )
  push!(dw3, "distinct 3-grams =" => length(dw3) )
  save_dict1_u_by2_desc(dw3, "words3gram.tsv")

  total = sum(collect(values(dw4)))
  push!(dw4, "total 4-grams =" => total )
  push!(dw4, "distinct 4-grams =" => length(dw4) )
  save_dict1_u_by2_desc(dw4, "words4gram.tsv") 

  total = sum(collect(values(dw5)))
  push!(dw5, "total 5-grams =" => total )
  push!(dw5, "distinct 5-grams =" => length(dw5) )
  save_dict1_u_by2_desc(dw5, "words5gram.tsv") 

  total = sum(collect(values(dw6)))
  push!(dw6, "total 6-grams =" => total )
  push!(dw6, "distinct 6-grams =" => length(dw6) )
  save_dict1_u_by2_desc(dw6, "words6gram.tsv") 

  total = sum(collect(values(dw7)))
  push!(dw7, "total 7-grams =" => total )
  push!(dw7, "distinct 7-grams =" => length(dw7) )
  save_dict1_u_by2_desc(dw7, "words7gram.tsv") 
  
  println("ngram tsv files saved")
  t1 = time()
  println("total time ",round((t1-t0)/3600,digits=3)," hours")
end  



function ppl2(datafile)
  d_joint = load_dict1_u(datafile); d_cond = Dict(); d_joint_sum = 0
  d_k_left = Dict(" a"=>1,"fg"=>4); empty!(d_k_left); d_k_left_sum = 0; d_cond_sum = 0
  
  for kj in keys(d_joint)
    d_joint_sum += get(d_joint, kj, 0) 
    #if length(kj) == 1
    #  k_left = ""
    #  d_k_left[ k_left ] = get(d_k_left, k_left, 0) + get(d_joint, kj, 0) 
    #else k_left = kj
    k_left = join(split(kj,"")[1:end-1]) 
    d_k_left[ k_left ] = get(d_k_left, k_left, 0) + get(d_joint, kj, 0) 
    #end
  end
  save_dict1_u_by1_desc(d_k_left, "d_k_left.tsv")
  
  for kj in keys(d_joint)
    k_left = join(split(kj,"")[1:end-1]) 
    d_cond[ kj ] = round(log2(get(d_joint, kj, 0) / get(d_k_left, k_left,0)), digits=5)
    #d_cond[ kj ] = get(d_joint, kj, 0) / get(d_k_left, k_left,0)
  end
  
  for kc in keys(d_cond)
    d_cond_sum += get(d_cond, kc, 0)
  end
  n = length(d_cond)
  
  save_dict1_u_by2_desc(d_cond, "d_cond_desc.tsv")
  println("d_joint_sum=", d_joint_sum, "   d_cond_sum=", d_cond_sum, "  n=", length(d_cond))
  println("    H=d_cond_sum/n=", d_cond_sum/n)
  println("2^(-H)=", 2^( -d_cond_sum/n ),"   l=", length(d_cond) )
end  



function insert_sym()
  io = open("enwiki_1M.txt","r")
  r = read(io,String);  s = collect(r); text = ""
  io = IOBuffer()
  for i in 1:length(s)
    #text = text * s[i] * " "
    text = s[i] * " "
    print(io, text )
  end  
  #  write( "a_ru_1M_sym.txt", text)
  text = String(take!(io));  write("enwiki_1M_sym.txt", text );  # close(fid)
end



function remove_syms(input_file, valid_syms)
    # remove_syms("webtext2_1M.txt", "ascii_87_sym.tsv")
    # valid_syms = "ascii_87.tsv"  input_file = "webtext2_100K.txt"
    t0 = time()

    output_file = split(input_file, ".")[1] * "_nosym.txt"
    valid = load_dict1_u(valid_syms)
    valid = Dict(collect(k)[1]=>v  for (k,v) in pairs(valid))
    push!(valid,'\n'=>1); #push!(valid,'\n'=>1)
    io_read = open(input_file,"r")

    c_input = filesize(input_file); i=0; chunk = 100_000_000
    c_output = 0
    #@time filter!(x->(x in collect(keys(valid))),s) ; l_output = length(s)
    io = IOBuffer() #maxsize=l_input + 100_000_000)
    while !eof(io_read)
        i+= 1 
        c = read(io_read, Char)
          if haskey(valid, c)
            print(io, c); c_output +=1
          else   
            print(io, " "); c_output +=1
          end
          if i%chunk == 0
            t_ = chunk * ((time() - t0) / i)
            t_ = round(t_, digits=4)
            println("syms=", i, " from ", c_input, 
                    " %=", round(100*i/c_input, digits=2), "  t_chunk=", t_); 
                    flush(stdout)
          end
        #end 
    end     
    text = String(take!(io))
    l_output = length(text)
    #text = join(s); 
    removed = c_input - c_output
    write(output_file, text );  # close(fid)
    println("removed =", removed, " from ", c_input, "  ",(removed/c_input)*100, " %" )
    println(output_file, " saved")
    t1 = time()
    println("total time ",round((t1-t0)/3600,digits=3)," hours")
end
  
  
  
function replace_chars(input_file)
      # replacement table -> "_chars2replace.tsv"
      # replace column2 => column3
      t0 = time()
      dict = load_chars2replace("_chars2replace.tsv")
      output_file = split(input_file, ".")[1] * "_chars_replaced.txt"
      io = IOBuffer()   
      fid = open(input_file)
          while !eof(fid)
              line = readline(fid)
              for (k,v) in pairs(dict)  
                  line = replace(line, k=>v)
              end
              line = replace(line, Char(769)=>"") # remove acute
              print(io, line * '\n')    
          end  
      text = String(take!(io))
      write(output_file, text)
      t1 = time()
      println(length(dict)," chars in the list of replacement")
      println("total time ", round((t1-t0)/3600, digits=3)," hours")
end
  


function load_chars2replace(chars2replace)
    # dict = load_chars2replace("_chars2replace.tsv")
    # tsv format:  Char	tab 005F	tab 0020
    dict = Dict()
    fid = open(chars2replace)
    while !eof(fid)
        line = readline(fid) # line = "_\t005F\t0020..."
        if !startswith(line, "#") && length(line) > 3
            key = Char( parse(UInt32, (split(line,'\t')[2]), base=16) ) 
            val = Char( parse(UInt32, (split(line,'\t')[3]), base=16) ) 
            dict[key] = val
        end
    end
    close(fid)
    println(length(dict)," chars in the list of replacement")
    return dict
end  



function replace_substrings(input_file, replace_tsv)
    # input_file="librusec.csv"  replace_tsv="_string2replace.tsv"
    # replace_substrings("librusec10_sentences.txt", "_string2replace.tsv")
    # multiple spaces -> replace(text, r"\s+" => " ")
    t0 = time()
    output_file = split(input_file, ".")[1] * "_replaced.txt"  
    lines_in = 0; lines_out = 0; chunk = 1_000_000
    datafile="_string2replace.tsv"
    dict = Dict()
    fid = open(replace_tsv)
    while !eof(fid)
        line = readline(fid)
        if !startswith(line, "#") && length(line) > 3
            oldstring = split(line, "\t\t")[1] # oldstring="003F\t002E" == "?."
            oldchars = split(oldstring, '\t') 
            key = join(map(x->Char( parse(UInt32, x, base=16) ), oldchars) ) # 003F -> "?"        
            newstring = split(line, "\t\t")[2] 
            newchars = split(newstring, '\t') 
            val = join(map(x->Char( parse(UInt32, x, base=16) ), newchars) ) 
            dict[key] = val
        end
    end
    
    kv = sort(collect(pairs(dict)), rev=true)
    println(length(kv)," substrings to replace")
    fid = open(input_file)
    lines_src = countlines(input_file) ; # filesize(input_file)
    io = IOBuffer()
    replaced = 0; replacement = 0
        while !eof(fid)
            line = readline(fid);   lines_in +=1;
            if length(line) > 0
                l_in = length(line)
                for i in 1:length(kv)
                    # sequentially from longest to shortest substring (not greedy)
                    line = replace(line, kv[i][1] => kv[i][2]) 
                    line = replace(line, r"\s+" => " ") # spaces
                end    
                l_out = length(line)
                print(io, line * "\n"); lines_out += 1        
                replaced += (l_in - l_out)

                if lines_out%chunk == 0
                    t_ = chunk * ((time() - t0) / lines_out)
                    t_ = round(t_, digits=3)
                    done = round(100*(lines_in / lines_src), digits=2)
                    println(done,"%  lines_out=", lines_out, "  t_chunk=", t_, "  replaced~", replaced)
                    flush(stdout)
                end
            end 
        end  
    text = String(take!(io)) 
    write(output_file, text) 
    close( fid )
    println("lines in=", lines_in, "    lines out=", lines_out);
    println(replaced, " ~chars replaced")
    println(output_file, " saved")
    t1=time()
    println("total ", round((t1-t0)/3600, digits=3)," hours")
end



function librusec_dir(input_dir)
  # librusec_dir(pwd()) input_dir=pwd()
  @assert isdir(input_dir)
  files = readdir(input_dir)
  files_out = 0
  println("files in dir=", length(files))
  level = 0.1
  for i in 1:length(files)
    input_file = input_dir * "/" * files[i]
    if split(input_file, ".")[2] == "csv"
      remove_by_stat(input_file, invalid_chars, level)
      files_out +=1
    end
  end
  println("csv files processed=", files_out)
end



function remove_by_stat(input_file, invalid_chars, level)
  # remove_by_stat(input_file) input_file = "librusec.csv"
  # valid_chars = "ascii_valid.tsv"
  # invalid_chars = "_invalid_chars.tsv"
  t0=time()
  #output_file = split(input_file, ".")[1]*"_"*string(level)*".txt" 
  output_file = split(input_file, ".")[1] * "_valid" * ".txt" 
  output_file2 = split(input_file, ".")[1] * "_invalid" * ".txt" 
  lines_in = 0; lines_out = 0; lines2_out = 0; chunk = 1_000_000 
  fractions = 0; lines_long = 0
  level = 0.001
  #valid = load_dict1_u(valid_chars)
  #valid = Dict(collect(k)[1]=>v  for (k,v) in pairs(valid))
  #valid = keys(valid)
  #push!(valid,'\n'=>1)
  invalid = load_dict1_u(invalid_chars)
  invalid = Dict(collect(k)[1]=>v  for (k,v) in pairs(invalid))
  invalid = keys(invalid)
  #push!(valid,'\n'=>1)

  fid = open(input_file)
  lines_src = countlines(input_file)
  io = IOBuffer(); io2 = IOBuffer(); #level = 0.15 # 5% punctuation
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    if length(line) > 0
      chars = collect(line); lines_long +=1
      #filtered = filter!(x->(x in valid), chars)
      filtered = filter!(x->(x in invalid), chars)
      #fraction = length(filtered) / length(line)
      #fractions += fraction
      #if fraction > level 
      #println("invalid chars in line=", length(filtered))
      if length(filtered) < 10 
        print(io, line * '\n'); 
        lines_out += 1
      else    
          println(length(filtered), " line=", lines_in)
          print(io2, line * '\n'); 
          lines2_out += 1
      end  
      #if lines_out%chunk == 0
      #  t_ = chunk * ((time() - t0) / lines_out)
      #  t_ = round(t_, digits=3)
      #  done = round(100*(lines_in / lines_src), digits=2)
      #  println(done,"%  lines_out=", lines_out, "  t_chunk=", t_); flush(stdout)
      #end
    end
  end 
  text = String(take!(io))
  write(output_file, text) 
  text2 = String(take!(io2)) 
  write(output_file2, text2) 
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println("lines in=", lines_in, "    lines2 out=", lines2_out);
  println(output_file, " saved")
  println("fraction avg=", fractions/lines_in)
  println("fraction avg in long lines=", fractions/lines_long)
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function calc_pmi(input_file, smooth=false)
    text = read(input_file, String)
    words = split(lowercase(text))

    w_count = Dict{String, Int64}()
    for word in words
        w_count[word] = get(w_count, word, 0) + 1
    end     

    vocab = length(w_count)

    if smooth
        sm = log(vocab) # max sm ~ vocab/10 
        println("smoothing k=", sm)
    end

    total_words = length(words)
    
    output_file = split(input_file, ".")[1] * "_vocab.tsv"
    save_dict1_u_by2_desc(w_count, output_file)
    
    w_pair_count = Dict{Tuple{String, String}, Int64}()
    for i in 1:length(words)-1
      w1 = words[i]
      w2 = words[i+1]
      w_pair_count[(w1,w2)] = get(w_pair_count, (w1,w2), 0) + 1
    end
    
    output_file = split(input_file, ".")[1] * "_pair.tsv"
    save_dict1_u_by2_desc(w_pair_count, output_file)
    
    ppmi_scores = Dict{Tuple{String, String}, Float64}()
    
    for i in 1:length(words)-1
        w1 = words[i]
        w2 = words[i+1]     
        total = length(words)
        
        if smooth
          p1 = (w_count[w1]+sm)/total
          p2 = (w_count[w2]+sm)/total
          p1p2 = (w_pair_count[(w1, w2)]+0)/total
        else
          p1 = w_count[w1]/total
          p2 = w_count[w2]/total
          p1p2 = w_pair_count[(w1, w2)]/total
        end  
          
        pmi_score = log( p1p2 / (p1 * p2) )
        # log2( w_pair_count[(w1,w2)] * total / w_count[w1] * w_count[w2])
        
        #ppmi_score = max(pmi_score, 0)
        ppmi_score = pmi_score
        
        freqw1w2 = get(w_pair_count, (w1, w2), 0)
        
        freqw1 = get(w_count, w1, 0)
        keyw1 = rpad(w1 * "(" * string(freqw1) * ")", 18)
        
        freqw2 = get(w_count, w2, 0)
        keyw2 = rpad(w2 * "(" * string(freqw2) * ")", 18)
        
        key = (keyw1, keyw2*"  "*"["*string(freqw1w2)*"]")
        ppmi_scores[key] = ppmi_score
    end   

    output_file = split(input_file, ".")[1] * "_pmi. tsv"
    save_dict1_u_by2_desc_io(ppmi_scores, output_file)
    println("vocab=", vocab, "  total_words=", length(words))
end


# Пример использования функции
# text = "this is a long text with many words and some repeated words"
# window_size = 2
#smooth = true

# pmi_matrix = compute_pmi(text, window_size, smooth)

#using StatsBase


function compute_pmi2(text::AbstractString, window_size::Int, smooth::Bool)
    # Разбиваем текст на слова
    words = split(text)
    
    # Создаем словарь для подсчета частоты каждого слова
    word_counts = countmap(words)
    
    # Создаем список уникальных слов
    unique_words = unique(words)
    
    # Создаем пустую матрицу для хранения PMI значений
    pmi_matrix = zeros(Float64, length(unique_words), length(unique_words))
    
    # Создаем словарь для сопоставления слов с их индексами
    word_indices = Dict(word => i for (i, word) in enumerate(unique_words))
    
    # Вычисляем PMI для каждой пары слов
    for i in 1:length(words)-window_size
        word1 = words[i]
        word2 = words[i+window_size]
        
        # Проверяем, есть ли такие слова в словаре
        if haskey(word_counts, word1) && haskey(word_counts, word2)
            count_word1 = word_counts[word1]
            count_word2 = word_counts[word2]
            
            # Вычисляем частоту встречаемости пары слов
            cooccurrence_count = countmap(words[i:i+window_size])
            count_cooccurrence = get(cooccurrence_count, (word1, word2), 0)
            
            # Получаем индексы слов
            index_word1 = word_indices[word1]
            index_word2 = word_indices[word2]
            
            # Вычисляем PMI
            # pmi = log(count_cooccurrence / (count_word1 * count_word2))
            pmi = log((count_cooccurrence * length(words)) / (count_word1 * count_word2))
            
            # Применяем сглаживание Лапласа, если требуется
            if smooth
                pmi += log(length(unique_words))
            end
            
            # Записываем PMI значение в матрицу
            pmi_matrix[index_word1, index_word2] = pmi
        end
    end
    
    return pmi_matrix
end

# Пример использования функции
#text = "this is a long text with many words and some repeated words"
#window_size = 2
#smooth = false true

#pmi_matrix = compute_pmi2(text, window_size, smooth)
 



function jsonl2strings(input_file)
  # jsonl2strings("2005-06.jsonl") input_file = "2005-06.jsonl"
  #using JSON
  lines_in = 0; lines_out = 0 
  output_file = split(input_file, ".")[1] * "_strings.txt"
  lines = readlines(input_file)
  io = IOBuffer()
  for i in 3:4 #length(lines)
      line = JSON.parse(lines[i])["text"]
      print(io, "<sop>" * line * "<eop>")
    end
    text = String(take!(io))
    l_output = length(text)
    #text = join(s); 
    removed = l_input - l_output
    write(output_file, text );  # close(fid)
    println("removed =", removed, " from ", l_input, "  ",(removed/l_input)*100, " %" )
    println(output_file, " saved")
end



function json_tiny2strings_dir(input_dir)
    # json_tiny2strings(pwd()) input_dir=pwd()
    @assert isdir(input_dir)
    files = readdir(input_dir)
    files_out = 0
    println("files in dir=", length(files))
    for i in 1:length(files)
      input_file = input_dir * "/" * files[i]
      if split(input_file, ".")[2] == "json"
        json_tiny2strings(input_file)
        files_out +=1
      end
    end
    println("json files processed=", files_out)
end



function json_tiny2strings(input_file)
    # json_tiny2strings("01_tiny.json") input_file = "01_tiny.json"
    # {"story": "_____text here______", "instruction":
    output_file = split(input_file, ".")[1] * "_stories.txt"
    io_read = open(input_file,"r") 
    input_text = read(input_file, String)
    io_write = IOBuffer()

    # (?<=mystr1).*?(?=mystr2) select all between mystr1 and mystr2
    reg = r"(?<={\"story\": ).*?(?=\"instruction\":)"
    m = eachmatch(reg, input_text)
    stories = collect(m) 
    println("stories count=", length(stories))
    for story in stories
      cleaned1 = replace( replace(story.match, "story\": \""=>""), "\", \"instruction"=>"")
      cleaned2 = replace( cleaned1, "\","=>"")
      cleaned3 = replace( cleaned2, "\""=>"")
      cleaned4 = replace( cleaned3, "\\n"=>" ")
      cleaned5 = replace( cleaned4, "\\"=>"")
      print(io_write, cleaned5 * "\n")
    end
    text = String(take!(io_write))
    write(output_file, text )
    println(output_file, " saved")
end



function text_sym_stat(input_file)
    # separators = [ [' '];  ['.']; [',']; ['!']; ['?']; [';']; ['_']] 
    # text_sym_stat("webtext2_1M.txt") input_file="webtext2_small.txt" 
    # dict3 = Dict( (Int, Int) => "morpheme1  morpheme2 ...")
    # (m, freq)	=> "morphemes"; sort by freq; output_file="morphemes_.tsv"
    t0 = time()
    d_sym = Dict(); d_sym_code = Dict(); 
    fid = open(input_file)
    l = countlines(input_file); chunk = 1_000_000
    lines_in = 0; lines_out = 0 
    while !eof(fid)
      line = readline(fid, keep=true);   lines_in +=1;
    
      s = collect(line)
      for i in 1:length(s) 

        s1 = only(s[i]) 
        code = string(Int(s1)) # slow down 3 times
        code = Int(s1) # slow down 3 times
        d_sym[ s1 ] = get(d_sym, s1, 0) + 1
        # hex = "U+" * string(Int(s1), base=16, pad=4)
        # d_sym_code[ (s1, code, hex) ] = get(d_sym_code, (s1, code, hex), 0) + 1
        d_sym_code[ (s1, code) ] = get(d_sym_code, (s1, code), 0) + 1
      end    

      if lines_in%chunk == 0
        t_ = chunk * ((time() - t0) / lines_in)
        t_ = round(t_, digits=4)
        println("syms=", lines_in, " from ", l, 
                " %=", round(100*lines_in/l, digits=2), "  t_chunk=", t_); 
                flush(stdout)
      end

    end
    #@time r = read(io,String);  
    #@time s = collect(r);

    #delete!(d_sym, '\n'); delete!(d_sym, '\r'); 
    #delete!(d_sym, '\t'); delete!(d_sym, "")
    push!(d_sym,  input_file * "  total symbols =" => sum(collect(values(d_sym))) )
    push!(d_sym,  input_file * "  unique symbols =" => length(d_sym) )
    output_file = split(input_file, ".")[1] * "_sym_stat.tsv"
    save_dict1_u_by2_desc(d_sym, output_file)
    println("unique symbols = ", length(d_sym))
    println(output_file, " saved")
    push!(d_sym_code,  input_file * "  total symbols =" => sum(collect(values(d_sym_code))) )
    output_file = split(input_file, ".")[1] * "_sym_code_stat.tsv"
    save_dict21_u_by2_desc(d_sym_code, output_file)
    t1 = time()
    println("total time ",round((t1-t0)/3600,digits=3)," hours")
end  



function text_words_stat(input_file, threshold=0)
    # text_words_stat("webtext2_10M.txt", 20)
    # threshold only for final vocab truncate
    t0=time()
    output_file = split(input_file, ".")[1] * "_words.tsv" # _apos.tsv
    lines_in = 0; lines_out = 0 
    words_in = 0; words_out = 0
    d_words = Dict("abc"=>1); empty!(d_words); sizehint!(d_words, 100_000_000)
    io = IOBuffer()
    fid = open(input_file); chunk = 1_000_000
    lines_src = countlines(input_file)
    while !eof(fid)
      #line = readline(fid, keep=true);   lines_in +=1;
      line = readline(fid);   lines_in +=1;
      words = [""]
      words = split(line, " ");  
      words = map(lowercase, words)
  
      if length(words) > -1 #0 # not only \n
        s=""; lines_out +=1

        for i in 1:length(words)

          # all words without punctuation & special characters
          #if (m = match(r"""^(\p{L}\.)+$|\p{L}{1,}(\-\p{L}+)?(\-\p{N}+)?|[\-\p{Sc}]?
          #    [\p{N}\-:,./()]*\p{N}[%°']?""", words[i])) !== nothing
          #    s = string(m.match)
          #end

          # every substring between spaces
          s = words[i] 
          s = rstrip(s, '.')
          s = rstrip(s, ',')
          s = rstrip(s, '?')
          s = rstrip(s, '!')
          s = rstrip(s, ':')
          s = rstrip(s, '\"')
          
          s = rstrip(s, '.')
          s = rstrip(s, '?')
          s = rstrip(s, '!')
          
          s = lstrip(s, '\"')

          
          # words with apostrophe
          #s=words[i]
          #if !startswith(s,"\n") && contains(s, "'") &&
          #  !startswith(s,"'") && !endswith(s, "'")
          #  d_words[ s ] = get(d_words, s, 0) + 1
          #end  
          
          if !startswith(s,"\n")
            d_words[ s ] = get(d_words, s, 0) + 1
          end  
          
        end
      end      
              
      if lines_in%chunk == 0
        t_ = chunk * ((time() - t0) / lines_in)
        t_ = round(t_, digits=3)
        done = round(100*(lines_in / lines_src), digits=2)
        println(done," %  lines_out=", lines_out, "  t_chunk=", t_)
        flush(stdout)
      end

    end  
    println("length(d_words)=", length(d_words))
    delete!(d_words, "") # delete empty if s="" (empty lines) 
    total = sum(collect(values(d_words)))
    println("lines in=", lines_in, "    lines out=", lines_out)
    println("empty lines=", lines_in - lines_out)
    #push!(d_words,  output_file * "  total words =" => total )
    #push!(d_words, "total_wordforms =" => length(d_words) )
    save_dict1_u_by1_desc(d_words, output_file)
    println("wordforms=", length(d_words))
    println(output_file, " saved")
    #=
    output_file = "vocab_640000.tsv"
    save_dict1_u_by2_desc(d_words,output_file,lines_num=640000)
    println(output_file, " saved")
    
    output_file = "vocab_320000.tsv"
    save_dict1_u_by2_desc(d_words,output_file,lines_num=320000)
    println(output_file, " saved")       
    
    output_file = "vocab_160000.tsv"
    save_dict1_u_by2_desc(d_words,output_file,lines_num=160000)
    println(output_file, " saved")       

    output_file = "vocab_80000.tsv"
    save_dict1_u_by2_desc(d_words,output_file,lines_num=80000)
    println(output_file, " saved")       

    output_file = "vocab_40000.tsv"
    save_dict1_u_by2_desc(d_words,output_file,lines_num=40000)
    println(output_file, " saved")       
    =#       

    t1=time()
    println("total time ",round((t1-t0)/3600,digits=3)," hours")
    
    vocab_truncate = threshold #10000
    filter!(pair -> pair.second > vocab_truncate, d_words)
    total = sum(collect(values(d_words)))
    push!(d_words,  output_file * "  total words =" => total )
    push!(d_words, "total_wordforms =" => length(d_words) )
    
    #=
    output_file = split(input_file, ".")[1] * "_popular.tsv"
    println("popular words=", length(d_words))
    save_dict1_u_by2_desc(d_words, output_file)
    =#
    return output_file
end



function text_words_stat_simple(input_file, threshold=0)
  # text_words_stat_simple("1ts.txt", 1000_000)
  # text_words_stat("1ts.txt")
  # threshold only for final vocab truncate
  t0=time()
  output_file = split(input_file, ".")[1] * "_vocab.tsv"
  lines_in = 0; lines_out = 0
  words_in = 0; words_out = 0
  d_words = Dict("abc"=>1); empty!(d_words); sizehint!(d_words, 10_000_000)
  io = IOBuffer()
  fid = open(input_file); chunk = 1_000_000
  lines_src = countlines(input_file)
  while !eof(fid)
    #line = readline(fid, keep=true);   lines_in +=1;
    line = readline(fid);   lines_in +=1;
    # words = [""]
    words = split(line, " ");  
    # words = map(lowercase, words)

    if length(words) > -1 #0 # not only \n
      #s=""; 
      lines_out +=1

      for i in 1:length(words)
        s = words[i]
        d_words[ s ] = get(d_words, s, 0) + 1       
      end

    end      
            
    if lines_in%chunk == 0
      t_ = chunk * ((time() - t0) / lines_in)
      t_ = round(t_, digits=3)
      done = round(100*(lines_in / lines_src), digits=2)
      println(done," %  lines_out=", lines_out, "  t_chunk=", t_)
      flush(stdout)
    end

  end  
  println("length(d_words)=", length(d_words))
  delete!(d_words, "") # delete empty if s="" (empty lines) 
  total = sum(collect(values(d_words)))
  println("lines in=", lines_in, "    lines out=", lines_out)
  println("empty lines=", lines_in - lines_out)
  push!(d_words,  output_file * "  total words =" => total )
  push!(d_words, "total_wordforms =" => length(d_words) )



  save_dict1_u_by1_desc(d_words, output_file)
  println("wordforms=", length(d_words))
  println(output_file, " saved")
  #=
  output_file = "vocab_640000.tsv"
  save_dict1_u_by2_desc(d_words,output_file,lines_num=640000)
  println(output_file, " saved")
  
  output_file = "vocab_320000.tsv"
  save_dict1_u_by2_desc(d_words,output_file,lines_num=320000)
  println(output_file, " saved")       
  
  output_file = "vocab_160000.tsv"
  save_dict1_u_by2_desc(d_words,output_file,lines_num=160000)
  println(output_file, " saved")       

  output_file = "vocab_80000.tsv"
  save_dict1_u_by2_desc(d_words,output_file,lines_num=80000)
  println(output_file, " saved")       

  output_file = "vocab_40000.tsv"
  save_dict1_u_by2_desc(d_words,output_file,lines_num=40000)
  println(output_file, " saved")       
  =#       

  t1=time()
  println("total time ",round((t1-t0)/3600,digits=3)," hours")
  
  vocab_truncate = threshold #10000
  filter!(pair -> pair.second > vocab_truncate, d_words)
  total = sum(collect(values(d_words)))
  push!(d_words,  output_file * "  total words =" => total )
  push!(d_words, "total_wordforms =" => length(d_words) )
  
  
  output_file = split(input_file, ".")[1] * "_popular.tsv"
  println("popular words=", length(d_words))
  save_dict1_u_by2_desc(d_words, output_file)
  
  return output_file
end



function remove_by_words(input_file, valid_words)
    # remove_by_words("a_ru_10M.utf8", "valid_words.tsv") #"a_ru_10M_stat.tsv1000")    #"valid_words.tsv")
    # remove_by_words("webtext2_10M.txt", "vocab_60000.tsv")
    t0=time(); chunk = 1_000_000
    output_file = split(input_file, ".")[1] * "_removed_words.txt"
    lines_in = 0; lines_out = 0 
    words_in = 0; words_out = 0
    valid = load_dict1_u(valid_words) # "stop_en_bigram_510_upper.tsv")
    io = IOBuffer()
    fid = open(input_file)
    lines_src = countlines(input_file)
    while !eof(fid)
      line = readline(fid);   lines_in +=1;
      line_valid = ""
        if length(line) > 0
          words = split(strip(line), " "); words_in += length(words)
            if length(words) > 0
              for w in 1:length(words) 
                if haskey(valid, words[w]) || haskey(valid, lowercase(words[w])) || 
                  words[w] == "—" || words[w] == "." || words[w] == "," ||
                  occursin(r"^[0-9]{1,4}$", words[w])|| words[w] == ""  ||
                  (last(words[w], 1) == "." && haskey(valid, chop(words[w])) ) ||
                  (last(words[w], 1) == "?" && haskey(valid, chop(words[w])) ) ||
                  (last(words[w], 1) == "!" && haskey(valid, chop(words[w])) ) ||
                  
                  
                  last(words[w], 1) == "," || last(words[w], 1) == "." ||
                  last(words[w], 1) == "?" || last(words[w], 1) == "!" ||
                  last(words[w], 1) == ":" || last(words[w], 1) == ";" ||
                  last(words[w], 1) == ")" || last(words[w], 1) == ")," ||
                  first(words[w], 1) =="(" || first(words[w], 1) =="«" ||
                  last(words[w], 1) == "»" || last(words[w], 1) == "»," ||
                  last(words[w], 1) == "»." || words[w] == "" ||
                  last(words[w], 2) == "; " || last(words[w], 2) == ": "

                  
                  #println("word=",words[w])
                  #println("w=", w)
                  #=
                  if w != length(words) # NOT last word in line
                    #println(words[w])           
                    if  words[w] != "." #&& words[w] != "," && words[w] != ";" 
                      line_valid = line_valid * words[w] * " "; words_out += 1
                    else
                      line_valid = line_valid  #* words[w] * " "; words_out += 1  
                    end

                  else  # last word in line
                    line_valid = line_valid * words[w] #* " "; words_out += 1
                  end
                  =#
                  line_valid = line_valid * words[w] * " "; words_out += 1
                  #println("line_valid=",line_valid)
                else line_valid = "" # this keep empty lines 
                  break # words[w] is invalid
                end
                
              end
              line_valid =chop(line_valid) * "\n"
            end
          if length(line_valid) > 0  
          print(io, line_valid); lines_out += 1
          end
        end  
    
        if lines_in%chunk == 0
          t_ = chunk * ((time() - t0) / lines_in)
          t_ = round(t_, digits=3)
          done = round(100*(lines_in / lines_src), digits=2)
          println(done, "%  lines_out=", lines_out, "  t_chunk=", t_)
          flush(stdout)
        end

    end 
    text = String(take!(io))
    write(output_file, text) 
    removed = words_in - words_out
    println("lines in=", lines_in, "    lines out=", lines_out);
    println("total words=", words_in, "    removed=", removed,  "  ",(removed/words_in)*100, " %")
    println("vocab file=", valid_words)
    println(output_file, " saved")
    t1=time()
    println("total time ",round((t1-t0)/3600,digits=3)," hours")
end



function remove_by_words_simple(input_file, valid_words)
  # remove_by_words("a_ru_10M.utf8", "valid_words.tsv") #"a_ru_10M_stat.tsv1000")    #"valid_words.tsv")
  # remove_by_words("webtext2_10M.txt", "vocab_60000.tsv")
  t0=time(); chunk = 1_000_000
  output_file = split(input_file, ".")[1] * "_rm.txt"
  lines_in = 0; lines_out = 0 
  words_in = 0; words_out = 0
  valid = load_dict1_u(valid_words) # "stop_en_bigram_510_upper.tsv")
  io = IOBuffer()
  fid = open(input_file)
  lines_src = countlines(input_file)
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    line_valid = ""
      if length(line) > 0
        words = split(strip(line), " "); words_in += length(words)
          if length(words) > 0
            for w in 1:length(words) 
              if haskey(valid, words[w]) 
                
                line_valid = line_valid * words[w] * " "; words_out += 1
                #println("line_valid=",line_valid)
              else line_valid = ""  
                break # words[w] is invalid
              end
              
            end
            line_valid =chop(line_valid) * "\n"
          end
        if length(line_valid) > 0  # 0 keep empty lines, 1 removes
        print(io, line_valid); lines_out += 1
        end
      end  
  
      if lines_in%chunk == 0
        t_ = chunk * ((time() - t0) / lines_in)
        t_ = round(t_, digits=3)
        done = round(100*(lines_in / lines_src), digits=2)
        println(done, "%  lines_out=", lines_out, "  t_chunk=", t_)
        flush(stdout)
      end

  end 
  text = String(take!(io))
  write(output_file, text) 
  removed = words_in - words_out
  println("lines in=", lines_in, "    lines out=", lines_out);
  println("total words=", words_in, "    removed=", removed,  "  ",(removed/words_in)*100, " %")
  println("vocab file=", valid_words)
  println(output_file, " saved")
  t1=time()
  println("total time ",round((t1-t0)/3600,digits=3)," hours")
end



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
    regex = r"(?<=[^\s\.]{2}[.!\?])[\s\\]" # \s -> space ?<=
    #regex = r"(?<=[.!?])\s+(?=[A-Z])"
    regex = r"(?<=[^\s\.]{2}[.!\?])[!\"\?\"\.\")[\s\\]"
  elseif type == "phrases"
    #regex = r"(?<=[^\s\.]{2}[.,;:!\?])[!\"][\s\\]"
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
      if length(s)>0
        for i in 1:length(s)
          print(io, s[i] * '\n'); lines_out += 1 # restore removed trailing \n
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
  println("lines in=", lines_in, "    lines out=", lines_out);
  #println("total words=", words_in, "    removed=", removed,  "  ",(removed/words_in)*100, " %")
  println(output_file, " saved")
  inbytes = filesize(input_file); outbytes = filesize(output_file)
  println("input_file_size=", inbytes, "  output_file_size=", outbytes, " prop=", outbytes/inbytes)
  t1=time()
  println("total time ",round((t1-t0)/3600,digits=3)," hours")

end



function sentences_stat(input_file)
    # sentences_stat("webtext2_1M.txt")
    # using Plots, Histogram
    t0=time()
    output_file = split(input_file, ".")[1] * "_sent_stat.tsv"
    #output_file2 = split(input_file, ".")[1] * "_sent_hash_freq.tsv"
    #output_file3 = split(input_file, ".")[1] * "_sent_hash.tsv"
    output_file4 = split(input_file, ".")[1] * "_sent_length.tsv"
    #output_file5 = split(input_file, ".")[1] * "_sent_count_length.tsv"
    lines_in = 0; lines_out = 0 
    io = IOBuffer();   #io2 = IOBuffer()
    fid = open(input_file); chunk = 1_000_000
    lines_src = countlines(input_file)

    # hash("Sentence.") -> 0x23f8745ccd7baa51 -> UInt64
    # Int(0x23f8745ccd7baa51) -> 2591949527484967505
    # UInt64(2591949527484967505) -> 0x23f8745ccd7baa51
    d_sent_stat = Dict(("a")=>1); empty!(d_sent_stat);
    d_sent_length = Dict(("a")=>1); empty!(d_sent_length);
    #d_sent_count_length = Dict(("a", 1)=>1); empty!(d_sent_count_length);
    #d_sent_hash_freq = Dict((0x23f8745ccd7baa51,"a")=>1); empty!(d_sent_hash_freq);
    #d_sent_hash = Dict((0x23f8745ccd7baa51)=>1); empty!(d_sent_hash);
    
    while !eof(fid)
        line = readline(fid) # if keep=true -> dont remove trailing \n or \r\n
        lines_in +=1;
        h = hash(line)
        len = length(line)
        #          if len > 70
        #            suffix = " ... " * string(len - 70) * " more chars"
        #            line = first(line,70) * suffix
        #          end
        k_sent_stat = line * '\t' * string(len)
        k_sent_stat = line
        #k_sent_hash_freq = (h, line)
        #k_sent_hash = (h)
        k_sent_length = (line)
        #k_sent_count_length = 
        
        d_sent_stat[ k_sent_stat ] = get(d_sent_stat, k_sent_stat, 0) + 1
        d_sent_length[ k_sent_length ] =  len
        #d_sent_hash_freq[ k_sent_hash_freq ] = get(d_sent_hash_freq, k_sent_hash_freq, 0) + 1
        #d_sent_hash[ k_sent_hash ] = get(d_sent_hash, k_sent_hash, 0) + 1

        if lines_in%chunk == 0
          t_ = chunk * ((time() - t0) / lines_in)
          t_ = round(t_, digits=3)
          done = round(100*(lines_in / lines_src), digits=2)
          println(done,"%  lines_out=", lines_out, "  t_chunk=", t_)
          flush(stdout)
        end

    end  

    println("lines in=", lines_in, "    lines out=", lines_out);
    println("saving output files ...")
    save_dict1_u_by2_desc_io(d_sent_stat, output_file)
    #save_dict1_u_by2_desc(d_sent_hash_freq, output_file2)
    #save_dict1_u_by2_desc(d_sent_hash, output_file3)
    #save_dict1_u_by2_desc(d_sent_length, output_file4)
    save_dict1_u_by2_desc_io(d_sent_length, output_file4)
    
    #println("total words=", words_in, "    removed=", removed,  "  ",(removed/words_in)*100, " %")
    println(output_file, " saved")
    #println(output_file2, " saved")
    
    
    println("plotting ...")
    val = values(d_sent_length); max_val = maximum(values(d_sent_length))
    uval = unique(values(d_sent_length)); count_uval = length(uval)
    getindex(maximum(values(d_sent_length)))
    plt = histogram(collect(val), bins=1:length(uval))
    title = "Sentences count \n max_length=" * string(max_val) *
            "\ndistinct lengths=" * string(count_uval) 
    savefig(plt, "1ts_sentences_count.png")
    
    # using StatsBase            
    title = "Sentences length \n max_length=" * string(max_val) *
            "\ndistinct lengths=" * string(count_uval) 
    plt = histogram(collect(val), bins=1:length(uval), 
    title = title)
    savefig(plt, "1ts_sentences_lengths.png")
    #hist = fit(Histogram, collect(val), nbins=length(uval))
    #plot(hist)

    #data1d = rand(100)
    #histogram(data1d, bins=10, label="histogram plot", legend=:topleft, ylims=(0,20))
    #h = fit(Histogram, data1d, nbins=10)
    #plot!(h, seriestype=:steps, lw=3, lc=:blue, label="StatsBase histogram")

      savefig(plt, "sent_lengths_hist.png")
    #  println("sent_lengths_hist.png  saved")
    
    
    t1=time()
    println("total time ",round((t1-t0)/3600,digits=3)," hours")
      
    #return d_sent_length

end



function match_string(input_file)
  # match_string("webtext2_10M.txt")
  t0=time()
  output_file = split(input_file, ".")[1] * "_match.tsv"
  lines_in = 0; lines_out = 0 
  io = IOBuffer();   #io2 = IOBuffer()
  fid = open(input_file); chunk = 1_000_000
  lines_src = countlines(input_file)
  d_match = Dict("abc"=>1); empty!(d_match)
  while !eof(fid)
    line = readline(fid) # if keep=true -> dont remove trailing \n or \r\n
    lines_in +=1;
    #if contains(line, "._ ")
    #  print(io, line * '\n'); lines_out += 1
    #end

    regex1 = r"\w\.\w\."
    regex2 = r"\w\.\w"
    regex3 = r"\w\w\.\w"
    if (m1 = match(regex1, line)) !== nothing 
      #if (m = match(r"""^(\p{L}\.)+$|\p{L}{1,}(\-\p{L}+)?(\-\p{N}+)?|[\-\p{Sc}]?
      #  [\p{N}\-:,./()]*\p{N}[%°']?""", words[i])) !== nothing
      s1 = string(m1.match)      
      d_match[ s1 ] = get(d_match, s1, 0) + 1
    end
    
    if (m2 = match(regex2, line)) !== nothing 
      s2 = string(m2.match)      
      d_match[ s2 ] = get(d_match, s2, 0) + 1
    #s = string(m.eachmatch(regex, line))
    #s = join(m.captures)
    #    words = getfield.(eachmatch(regex, line), :match)
    #          print(io, join(words, "  ") * " || " *line * '\n'); lines_out += 1
    end  
    if lines_in%chunk == 0
      t_ = chunk * ((time() - t0) / lines_in)
      t_ = round(t_, digits=3)
      done = round(100*(lines_in / lines_src), digits=2)
      println(done,"%  lines_out=", lines_out, "  t_chunk=", t_)
      flush(stdout)
    end
  end  
  #  text = String(take!(io))
  #  write(output_file, text) # "input_filename._tmp"
  save_dict1_u_by2_desc(d_match, output_file)
  println("lines in=", lines_in, "    lines out=", lines_out);
end  



function remove_by_sentence(input_file, invalid_sentences)
  # remove_by_sentence("webtext2_10M.txt","webtext2_small_sent_hash.tsv")
  # remove_by_sentence("webtext2_small_test.txt","webtext2_small_sent_hash.tsv")
  # input_file = "webtext2_small.txt"
  # invalid_sentences = "webtext2_small_sent_hash.tsv"
  t0=time()
  output_file = split(input_file, ".")[1]*"_nosentences.txt" 
  lines_in = 0; lines_out = 0; lines2_out = 0; chunk = 100_000 
  # fractions = 0; lines_long = 0
  invalid = load_dict1_u_hash(invalid_sentences); println(invalid)
  invalid_k = keys(invalid); #println(invalid_k)
  #push!(valid,'\n'=>1); #push!(valid,'\n'=>1)

  fid = open(input_file)
  lines_src = countlines(input_file)
  io = IOBuffer(); io2 = IOBuffer()
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    if length(line) > 0

      if hash(line) in invalid_k 
          print(io2, line * '\n'); lines2_out += 1 # junk
        else    
          print(io, line * '\n'); lines_out += 1
      end  
      if lines_in%chunk == 0
        t_ = chunk * ((time() - t0) / lines_out)
        t_ = round(t_, digits=3)
        done = round(100*(lines_in / lines_src), digits=2)
        println(done,"%  lines_out=", lines_out, "  t_chunk=", t_); flush(stdout)
      end
    end
  end 
  text = String(take!(io)) 
  write(output_file, text) 
  text2 = String(take!(io2)) 
  write(output_file * "_2", text2) # junk
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println("lines in=", lines_in, "    lines2 out=", lines2_out);
  println(output_file, " saved")
  # println("fraction avg=", fractions/lines_in)
  # println("fraction avg in long lines=", fractions/lines_long)
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function eop(input_file)
  # eop("webtext2_small.txt") input_file = "1.txt"
  # eop("PG10000_text.txt") input_file = "PG10000_text.txt"
  t0=time()
    if Sys.iswindows()
          eol = "\r\n"
    else  eol = "\n"
    end  
  eol = "\n"
  output_file = split(input_file, ".")[1] * "_eop.txt" 
  #lines_in = 0; lines_out = 0 
  fid = open( input_file )
  text = read(input_file, String)
  text_eop = replace(text, eol * eol =>"<eop>") # "\r\n\r\n" =>"<eop>")
  t11=time(); println("LF => <eop> ", round((t11-t0)/3600, digits=3)," hours")
  text=[]

  text_eop_space = replace(text_eop, eol =>" ")
  t12=time(); println("eol =>  ", round((t12-t11)/3600, digits=3)," hours")
  text_eop=[]

  #text_eop_eos_lf = replace(text_eop_space, "<eop>" =>"<eop>\n")
  text_eop_eos_lf = replace(text_eop_space, "<eop>" =>"\n")
  t13=time(); println("eop => LF ", round((t13-t12)/3600, digits=3)," hours")
  text_eop_space=[]

  write(output_file, text_eop_eos_lf) # creates "input_file_eop.txt"
  close( fid )
  #println("lines in=", lines_in, "    lines out=", lines_out);
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function remove_punct(input_file)
  # remove_punct(input_file) input_file = "webtext2_10M.txt"
  t0=time()
  output_file = split(input_file, ".")[1] * "_punct.txt" 
  lines_in = 0; lines_out = 0; chunk = 1_000_000 
  fid = open(input_file)
  lines_src = countlines(input_file)
  io = IOBuffer()
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    if length(replace(line, " "=>"")) > 4
      if !all(ispunct, replace(line, " "=>"") ) &&
         !all(isnumeric, replace(line, " "=>"") ) &&
         !all(ispunct, replace( replace(line, isnumeric=>"") ," "=>"" ) )
         print(io, line * "\n"); lines_out += 1
      end 

      if lines_out%chunk == 0
        t_ = chunk * ((time() - t0) / lines_out)
        t_ = round(t_, digits=3)
        done = round(100*(lines_in / lines_src), digits=2)
        println(done,"%  lines_out=", lines_out, "  t_chunk=", t_)
        flush(stdout)
      end
      
    end
  end 
  text = String(take!(io)) # this reset and clears io
  write(output_file, text) # creates "input_file.tmp"
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function count_punct(input_file) # !!!!  WIP
  # remove_punct(input_file) input_file = "webtext2_10M.txt"
  t0=time()
  output_file = split(input_file, ".")[1] * "_punct.txt" 
  lines_in = 0; lines_out = 0; chunk = 1_000_000 
  fid = open(input_file)
  lines_src = countlines(input_file)
  io = IOBuffer()
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    if length(line) > 0
      if !all(ispunct, replace(line, " "=>"") ) &&
         !all(isnumeric, replace(line, " "=>"") ) &&
         !all(ispunct, replace( replace(line, isnumeric=>"") ," "=>"" ) )
         print(io, line * "\n"); lines_out += 1
      end  
      if lines_out%chunk == 0
        t_ = chunk * ((time() - t0) / lines_out)
        t_ = round(t_, digits=3)
        done = round(100*(lines_in / lines_src), digits=2)
        println(done,"%  lines_out=", lines_out, "  t_chunk=", t_)
        flush(stdout)
      end
    end
  end 
  text = String(take!(io)) # this reset and clears io
  write(output_file, text) # creates "input_file.tmp"
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function remove_substrings(input_file, junk_tsv)
  # input_file="webtext2_10M.txt"  junk_tsv="to_fix_40G.tsv"
  # remove_substrings("webtext2_10M.txt", "to_fix_40G.tsv")
  t0=time()
  output_file = split(input_file, ".")[1] * "_no_substr.txt" 
  lines_in = 0; lines_out = 0; chunk = 1_000_000
  junk = load_dict1_u(junk_tsv)

  fid = open(input_file)
  lines_src = countlines(input_file) ; # filesize(input_file)
  io = IOBuffer()
  removed = 0
    while !eof(fid)
      line = readline(fid);   lines_in +=1;
      if length(line) > 0
        l_in = length(line)
        for j in keys(junk)
          line = replace(line, j=>" ")
        end    
        l_out = length(line)
        print(io, line * "\n"); lines_out += 1        
        removed += (l_in - l_out)

        if lines_out%chunk == 0
          t_ = chunk * ((time() - t0) / lines_out)
          t_ = round(t_, digits=3)
          done = round(100*(lines_in / lines_src), digits=2)
          println(done,"%  lines_out=", lines_out, "  t_chunk=", t_, "  removed~", removed)
          flush(stdout)
        end

      end 
    end  
  text = String(take!(io)) # this reset and clears io
  write(output_file, text) # creates "input_file.tmp"

  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println(removed, " ~chars removed")
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
end



function sentences2lines(input_file)
  # sentences2lines("webtext2_small.txt")
  # sentences2lines("webtext2_1M_nosym.txt")
  # sentences2lines("webtext2_100M.txt")
  # ---------------------------------------
  # split paragraphs into sentences by ". "
  t0=time()
  output_file = split(input_file, ".")[1] * ".tmp" 
  lines_in = 0; lines_out = 0 
  #sentences_in = 0; sentences_out = 0
  io = IOBuffer()
  fid = open( input_file )
  while !eof(fid)
    line = readline(fid)  ;   lines_in +=1;
    if length(line) > 1 # "Ц" #Chars in line (no LF)
      sentences = split(line, ". "); #words_in += length(words)
      for s in 1:length(sentences)
        if s == length(sentences)
             new_line = sentences[s] * "\n"
        else new_line = sentences[s] * "." * "\n"
        end
        print(io, new_line); lines_out += 1
      end
    end  
  end 
  text = String(take!(io)) # this reset and clears io
  write(output_file, text) # creates "input_file.tmp"
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  #---------------------------------------- 
  # split paragraphs into sentences by "? "
  fid = open( output_file ) # *.tmp
  output_file = split(input_file, ".")[1] * ".tmp2"
  lines_in = 0; lines_out = 0
  while !eof(fid)
    line = readline(fid)  ;   lines_in +=1;
    if length(line) > 1 # "Ц" #Chars in line (no LF)
      sentences = split(line, "? "); #words_in += length(words)
      for s in 1:length(sentences)
        if s == length(sentences)
             new_line = sentences[s] * "\n"
        else new_line = sentences[s] * "?" * "\n"
        end
        print(io, new_line); lines_out += 1
      end
    end  
  end 
  text = String(take!(io))
  write(output_file, text) # creates "input_file.tmp2"
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  #----------------------------------------
  # split paragraphs into sentences by "! "
  fid = open( output_file ) # *.tmp2
  lines_in = 0; lines_out = 0
  output_file = split(input_file, ".")[1] * ".tmp3" 
  while !eof(fid)
    line = readline(fid)  ;   lines_in +=1;
    if length(line) > 1 # "Ц" #Chars in line (no LF)
      sentences = split(line, "! "); #words_in += length(words)
      for s in 1:length(sentences)
        if s == length(sentences)
          new_line = sentences[s] * "\n"
        else new_line = sentences[s] * "!" * "\n"
        end
        print(io, new_line); lines_out += 1
      end
    end  
  end 
  text = String(take!(io))
  write(output_file, text) # creates "input_file.tmp3"
  output_file = split(input_file, ".")[1] * "_lines.txt" 
  write(output_file, text) # creates "input_file.tmp3"
  close( fid )
  println("lines in=", lines_in, "    lines out=", lines_out);
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")
  #----------------------------------------------------------
  # split paragraphs into sentences by "." "  example:|about that." Although|
  # the same block of code
end



function pg_select()
  # id,title,author,authoryearofbirth,authoryearofdeath,language,downloads,subjects,type
  #using CSV 
  c=CSV.File("metadata.csv")
  c
  ids = c.id; lang = c.language
  #lang[2] ;ids[2]; #i = filter(lang == "['de']", c)
  copied = 0
    for i in 1:length(c)
      if lang[i] == "['en']"
        #print(ids[i])
        file_name = ids[i] * "_text.txt"
          try  
            cp(file_name, pwd() * "/en/" * file_name; force=true)
            copied +=1; println(i)
          catch 
            println("no file ", file_name)
            continue
          end  
      end
    end  
  println("copied=", copied,"   from ",length(c))
end



function books3_select(limit)
  # limit in MB
  copied = 0; dirsize = 0
  used_folders = []; used_files = []; done=false
  folder = readdir(pwd()) # i=2
  for i in 1:length(folder)
    if isdir(pwd() * "/" * folder[i]) && folder[i] != "tmp" && !done
      dirname = pwd() * "/" * folder[i]
      push!(used_folders, folder[i])
      files = readdir(dirname)
      for j in 1:length(files)
        if dirsize/1_000_000 < limit && !done
          try  
            src = dirname * "/" * files[j]
            dest = pwd() * "/tmp/" * files[j]
            cp(src, dest; force=true)
            dirsize += filesize(src) 
            copied +=1; #println(j, " j")
            push!(used_files, files[j])
          catch 
            println("no files in ", src)
            continue
          end  
          #println("copied=", copied," to ", )
          println("dirsize MB=", dirsize/1_000_000)
        else println("the limit=", limit ,"  has been reached")
             done = true 
            break 
        end
      end  
    end
  end  
  println("used folders=", length(used_folders),)
  println("used files=", length(used_files),)
  write("folders", string(used_folders))
  write("files", string(used_files))
  #return used_folders, used_files
end



function lang_detector()
  # using LanguageFinder 
  # input_file="webtext2_small.txt" "webtext2_100K.txt"
  t0=time()
  select = 16 # 20 lines selected
  ngram = 4 # 1,2,3,4, 0=&&(1,2,3,4)
  folder = readdir(pwd()  )  # * "/lang/"
  println("pwd=", pwd())
  println("select=", select, "  ngram=", ngram)
  println("files=", length(folder))
  langs_all = []; langstring = ""; eng = 0; oth = 0 # langs_allstring = ""
  copied = 0; chunk = 100
  L = LanguageFinder.LanguageFind
    for j in 1:length(folder)

      if isfile(pwd() * "/" * folder[j]) && filesize(pwd() * "/" * folder[j]) > 100
        input_file = pwd() * "/" * folder[j]
        #println(input_file)
        fid = open(input_file); l = filesize(input_file)
        step = Int(round(l/select)) 
        lang = ""; langs = []; langstring = ""

        for i in 1 : step : filesize(input_file) 
          i += step
          seek(fid, i) #    seek(fid, 1)
          line=readline(fid)
          if length(line) > 54
            #n=1
            #while !eof(fid)
            #  n +=1
            #  seek(fid, i +n)
            #  line=readline(fid)
            #  if length(line) > 54
            #    break
            #  end  
            #end  
              
            #else    
                try snippet = first(line, 50) 
                  lang = L(snippet, ngram).lang # 4-grams
                  push!(langs, lang)
                  langstring = langstring * " " * lang
                catch 
                  continue
                end               
          end
        end  
        
        if  count(x->(x=="en"), langs) > (0.55 * length(langs)) &&  #quorum =
            length(langs) >= 4
            detected = "EN "; eng +=1 
            src = pwd() * "/" * folder[j]
            dest = pwd() * "/en/" * folder[j]
            cp(src, dest; force=true)
            #dirsize += filesize(src) 
            copied +=1; #println(j, " j")
          else detected = "?? "; oth +=1
            src = pwd() * "/" * folder[j]
            dest = pwd() * "/other/" * folder[j]
            cp(src, dest; force=true)
        end 

        langstring =  detected * langstring * "  " * folder[j] * '\n'    
        println(length(langs))
        push!(langs_all, langstring )
        close( fid )

        if j%chunk == 0
          t_ = chunk * ((time() - t0) / j)
          t_ = round(t_, digits=3)
          println("scanned=", j, "  t_chunk=", t_)
          flush(stdout)
        end

      end  
      
    end
    report = ""    
    for i in 1:length(langs_all)
      report = report * string(langs_all[i]) #* '\n'
    end  
    summary1 = "EN=" * string(eng) * "  other=" * string(oth) * " \n" 
    summary2 = "select=" * string(select) * "  ngram=" * string(ngram) * " \n" 
    report = summary1 * summary2 * report 
    println("files=", length(langs_all))  
    println("english=", eng, "  other=", oth) 
    println("copied en=", copied, "  from ", length(folder)) # 1 
    write("langs_all", report)
    t1=time()
    println("total ",round((t1-t0)/3600, digits=3)," hours")
  #return langs_all #, quorum
end



function webtext_jsonl2text()
  # input_file = "2005.jsonl"
  t0=time()
  output_file = "all_jsonl.txt" 
  files_in = 0; lines_in = 0; lines_out = 0; chunk = 100_000 
  io = IOBuffer()
  for f in readdir(pwd())   #f=readdir(pwd())[3]
    if isfile(f) && split(f, ".")[2] == "jsonl"
      files_in +=1; println("files_in=", files_in)
      fid = open(f)
        while !eof(fid)
          line = readline(fid)
          d=JSON.parse(line) # "2005.jsonl")
          text = d["text"]
          lines_out +=1
          print(io,text)  #* "\n")
          if lines_out%chunk == 0
            t_ = chunk * ((time() - t0) / lines_out)
            t_ = round(t_, digits=3)
            progress = 100* (files_in / length( readdir(pwd()) ) )
            println(progress, "%  lines_out=", lines_out, "  t_chunk=", t_); flush(stdout)
          end
        end 
    close(fid)     
    end
  end
  text = String(take!(io)) # this reset and clears io
  write(output_file, text) # creates "input_file.tmp"
  println("files in=", files_in, "    lines out=", lines_out);
  println(output_file, " saved")
  t1=time()
  println("total ",round((t1-t0)/3600, digits=3)," hours")  
end



function dot_inside_word(input_file)
  output_file = split(input_file, ".")[1] * "_dots.txt"
  lines_in = 0; lines_out = 0 
  words_in = 0; words_out = 0
  two_chars = Dict()
  io = IOBuffer()
  fid = open(input_file)
  while !eof(fid)
    line = readline(fid);   lines_in +=1;
    if length(line) > 2
      words = split(line, " "); #words_in += length(words)
      if length(words) > 0
        for w in 1:length(words)
          if contains(words[w], ".") #&& !endswith(words[w], ".")
          two_chars[words[w]] = get(two_chars, words[w], 0) + 1
          end  
          #if length(words[w]) == 2 && !all(isdigit, words[w])
          #  two_chars[words[w]] = get(two_chars, words[w], 0) + 1
          #end
        end  
      end
    lines_out += 1
    end  
  end 
  save_dict1_u_by2_desc( two_chars, output_file )
  #println("lines in=", lines_in, "    lines out=", lines_out);
  #println("total words=", words_in, "    removed=", words_in - words_out)
end



""" ~ StatBase.countmap replacement to avoid StatBase pkg add requirement"""
#=function countmap(v::Vector)::Dict
    d = Dict(k => 0 for k in unique(v))
    for x in v
        d[x] += 1
    end
    return d
end 
=# 



function inside_memory()
    @time a = rand(100,100)
    function inside_memory2()
      @time a = rand(100,100)
    end  
    function inside_memory3()
      @time a = rand(100,100)
    end  
end  



function memory()
    # on windows
    s = String(read(`tasklist /FI "PID eq $(getpid())" /NH /FO csv`))
    println(s)
    # parse(Int, replace(match(r"""([^"]+)"[^"]+$""", s).captures[1], r"[.,K]" => ""))
    parse(Int, replace(match(r"""([^"]+)"[^"]+$""", s).captures[1], r"[.,K]" => ""))
end



function get_mem_use()
    # on linux
    f = open( "/proc/self/stat" )
    s = read( f, String )
    vsize = parse( Int64, split( s )[23] )
    mb = Int( ceil( vsize / ( 1024 * 1024 ) ) )
    gb = Int( ceil( vsize / ( 1024 * 1024 * 1024 ) ) )
    return gb
end



function catch_outofmemory()
  println("free memory ", Sys.free_memory()/2^30) 
    try
      rand(10^15) # A=zeros(100000, 100000) 
      catch OutOfMemoryError
      free = Sys.free_memory()/2^30
      println("attempting allocate more then ", free)
      println("free memory ", free) 
    end
end  



function remove_unwanted_symbols(s::String, allowed_set::Set{Char})
  buffer = IOBuffer()
  for c in s
    if c in allowed_set
      write(buffer, c)
    else
      write(buffer, ' ')
    end
  end
  return String(take!(buffer))
end



function compare_dicts()
    d1 = load_dict1_u("1_gen.tsv")
    d2 = load_dict1_u("vocab.tsv")

    same = intersect( collect(keys(d1)), collect(keys(d2)) )
    #same = intersect( collect(keys(d2)), collect(keys(d1)) )
    
    #diff = setdiff( collect(keys(d1)), collect(keys(d2)) )
    #diff = setdiff( collect(keys(d2)), collect(keys(d1)) )

    #diff = symdiff( collect(keys(d1)), collect(keys(d2)) )
    #printl(same)  
    println("l1=", length(d1))
    println("l2=", length(d2))
    println("common elements=", length(same) ) 
end



function clusters(datafile)
  lines = readlines(datafile); # l = length(lines)
  #lines = readlines("log4000-2048-6-75clust")log4000-4096-3-6
  words = ["(пришел)","(пришёл)","(пришел","(пришёл","пришел","пришёл" ] 
  #words = ["(париж","(украин","сказал", "(рентгенов", "(университет", "(фсб" , 
  # "(пришел", "(ответил", "(поэт)","(майор)",
  # "(пушкин", "(роснано","(британский)", "(путин)",
  #"(февраль", "(миллион", 
  #"(долларов)","(долларов","долларов","(доллар)","(доллар",#]
  #"(доллара)","(доллара","доллара","(долларов)","(долларов","долларов",
  #"(доллару)","(доллару","доллару",
  #"(строительства","(восточной","(войсками", 
  #"(долларам)","(долларам","долларам"]
  
  f=open("clusters.txt","w")
  write(f, datafile * "\n" * string(words) * "\n" * "--------------\n")
  for k in 1:length(words) 
    println(words[k])
    write(f, words[k] * "\n")
    for i in 1:length(lines)
      if contains(lines[i], words[k])
        write(f, lines[i] * "\n")
      end  
    end
    write(f, "--------------\n")
  end
  close(f)
end



function subtract_words(input_file,output_file,stop_words_file)
  # remove all stopwords occurences
  fid_in  = open(input_file)
  dirty = read(fid_in)

  v = readlines(stop_words_file)
  #v_spaces = Array{String}(length(v),1)
  v_spaces = Array{String}(undef,length(v))
  #v_comma = Array{String}(length(v),1)
  v_comma = Array{String}(undef,length(v))
  #v_dot_after = Array{String}(length(v),1)    # " one."
  v_dot_after = Array{String}(undef,length(v))    # " one."
  #v_dot_before = Array{String}(length(v),1)   # ".one"
  v_dot_before = Array{String}(undef,length(v))   # ".one"
  #v_LF_before = Array{String}(length(v),1)   # "\n"first word on new line
  v_LF_before = Array{String}(undef,length(v))   # "\n"first word on new line
  #v_LF_before_space_after = Array{String}(length(v),1)   # "\n"first word on new line
  v_LF_before_space_after = Array{String}(undef,length(v))   # "\n"first word on new line
  close(fid_in)
  println("removing ",length(v)," stops-words")
  println("source length = ",length(dirty))

  subtracted = dirty
    for i in 1:length(v)
      #v[i] = replace(v[i],"\r\n","")
      v[i] = replace(v[i],"\r\n"=>"")

      v_spaces[i] = " " * v[i] * " "                    # " the "
      subtracted = replace(subtracted,v_spaces[i]=>" ")

      v_comma[i] = " " * v[i] * ","                     # " the,"
      subtracted = replace(subtracted,v_comma[i]=>" ")

      v_dot_after[i] = " " * v[i] * "."                 # " the."
      subtracted = replace(subtracted,v_dot_after[i]=>" ")

      v_dot_before[i] = "." * v[i] * " "                # ".the "
      subtracted = replace(subtracted,v_dot_before[i]=>" ")

      v_LF_before[i] = "\n" * v[i] * ","
      subtracted = replace(subtracted,v_LF_before[i]=>"\n ")

      v_LF_before_space_after[i] = "\n" * v[i] * " "
      subtracted = replace(subtracted,v_LF_before_space_after[i]=>"\n ")

      if i%100 == 0
              #println("count=",i)
              #println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
      end
    end
  println("cleaned length = ",length(subtracted))
  fid_out = open(output_file,"w")
  write(fid_out,subtracted)
  close(fid_out)
  return 0
end



function prepare_stops(input_words,output_words)
  # Adds stop-words starting in uppercase, e.g. if exists "the" adds "The"
  fid_in  = open(input_words)
  lower = readlines(fid_in)
  close(fid_in)
  println("adding uppercases ...")
        for i in 1:length(lower)
            lower[i] = join([lower[i],ucfirst(lower[i])])
        end
  upper =  lower
  fid_out = open(output_file,"w")
  write(fid_out,upper)
  close(fid_out)
  return 0
end



function create_sparse_efficient(k)
    I, J, V = Int[], Int[], Float64[]
    for i = 1:k
        idxs = (i-1)*2 + 1:i*2
        for i in idxs, j in idxs
            push!(I, i)
            push!(J, j)
            push!(V, rand())
        end
    end
    return sparse(I,J,V)
end



function create_sparse_inefficient(k)
    # Indexing operations,especially assignment, are expensive, 
    # when carried out one element at a time
    M = spzeros(2*k,2*k)
    for i = 1:k
      D = (i-1)*2 + 1:i*2
      M[D,D] = randn(2,2)
    end
    return M
end



function lp(v)
  if v==1
    pw = [-10.86, -4.08, -8.27, -0.31, -6.10, -1.29, -20.88, -10.03, -6.15]
  elseif v==2 ? pw = [-10, -10,-10,-10,-10,-10,-10,-10,-10, -10] : 
    pw = [-20, -10,-10,-10,-10,-10,-10,-10,-10, -10]
  end  
  lp = 0
  for i in 1:length(pw)
    lp += (pw[i])
  end
  return length(pw), lp, lp / length(pw), 2^(-lp / length(pw) )
end



function ppl_golem(file)
  # file.tsv -> 'w'=>1
  # golem_chars.tsv
  # PPL = -sum( P(k) * log(P(k)) ) 
  d = load_dict1_u(file)
  sum_freq = 0
  PPL = 0.0  # Perplexity
  ENT =0.0; ENT2 =0  # Entropy
  for k in keys(d)
    sum_freq += get(d, k, 0)
  end
  println("sum_freq=", sum_freq)
  for k in keys(d)
    cnt = get(d, k, 0)
    freq = cnt/sum_freq
    ENT -= freq * log2(freq)
    ENT2 += freq * log2(freq)
  end   
  PPL = 2 ^ ENT
  println("length(d)=", length(d))
  println("PPL=", PPL)
  println("ENT=", ENT)
  println("Length/PPL=", length(d) / PPL )
  return # ENT, ENT2 #PPL 
end



function yadialog()
  # context_id, context_2, context_1, context_0, reply_id, reply, label,confidence
  # context_id,        s2,        s3,        s4,       s5,    s6, label,confidence
    fid = open( "train.tsv" )
    io = IOBuffer(); lines_in = 0; lines_out = 0 
      while !eof(fid)
        line = readline(fid);   lines_in +=1;  
        (s1,s2,s3,s4,s5,s6,s7,s8) = split(chomp(line),'\t')
        new_id = id = s1; newline =""
        while (new_id == id) && (!eof(fid))
          score = 0
          line = readline(fid);   lines_in +=1;
          (s1,s2,s3,s4,s5,s6,s7,s8) = split(chomp(line),'\t')
          new_id = s1; new_score = parse(Float64,s8)
            if (s2 !="") && (s3 !="") # save only 4-replicas
              if (s7=="good") && (new_score > score)
                  #println(new_score) # = s8
                  newline = s2 * '\t' * s3 * '\t' * s4 * '\t' * s6 * " \n" #;lines_out +=1
              end
            end  
        end
        print(io, newline );  lines_out +=1
      end
    text = String(take!(io));  write("text.txt", text );   close(fid)
    return lines_in, lines_out
end



function pgn_clean()
  dump1 = read("pgn_raw.pgn", String)
  dump2 = replace(dump1, "\r\n\r\n" => "^^")
  dump3 = replace(dump2, "\r\n" => " ")
  dump4 = replace(dump3, "^^" => "\n")
  dump5 = replace(dump4, "1-0" => "")
  dump6 = replace(dump5, "0-1" => "")
  dump7 = replace(dump6, "1/2" => "")
  dump8 = replace(dump7, "+" => "")
  dump9 = replace(dump8, "#" => "")
  lines = split(dump9, "\n"); games = 0
  lines_for_sorting=[""]; #sizehint!(lines_for_sorting, 20_000_000) 
  f=open("pgn_cleaned.txt","w"); #fs=open("Adam-s.txt","w");ws = "";
  for n in 1:length(lines)
      if !startswith(lines[n], '[')
          splitted = split(lines[n],'.') 
          clean = ""; move =  ""
              for s in 1:length(splitted) 
                    if s == 1           # 1.
                    move = ""  
                    elseif 1 < s < 10   # 1.{ d4 Kf6 2}.
                    move = chop(splitted[s], head=1, tail=1 )  
                    elseif  s < 99
                      if s == length(splitted)
                        move = chop(splitted[s], head=1, tail=1 )
                      else
                        move = chop(splitted[s], head=1, tail=2 )  # ""
                      end  
                    elseif  s >= 99
                    move =  ""
                    end  
                    clean = clean * move 
              end  
          push!(lines_for_sorting, clean * " \n") 
          
          write(f, clean * " \n"); games+=1;
        end
  end
  sorted = sort(lines_for_sorting); 
  io = IOBuffer()   
  for w in 1:length(sorted)
    print(io, sorted[w])
  end 

  text_buffer = String(take!(io))
  write("pgn_cleaned_sorted.txt", text_buffer)
  close(f);   println("games =", games)
end



function pgn_stat()
    io = open("pgn_cleaned.txt","r")
    r = read(io,String);  #s = collect(r);
    moves = split(r, " ")
    stat = Dict(); #d2 = Dict();  
  for i in 1:length(moves) 
    move = moves[i] 
    if startswith(move,"\n") move = replace(move, "\n" =>"") end #chop(move, head=1 ) end
    if !isempty(move) && !startswith(move," ")
      stat[ move ] = get(stat, move, 0) + 1
  #    if Char(s[i+1]) in alphabet
  #      s12 = s[i] * s[i+1]
  #      d12[s12] = get(d12, s12, 0) + 1
  #    end   
    end  
  end  
        save_dict1_u(stat, "pgn_stat.tsv"); println(length(moves))
        #save_dict1_u(d12, "d12.tsv")
end  


      
function pick_rand_strings()

  lines = readlines("result_full.txt")
  l = length(lines)

  f=open("samples.txt","a+")

  for n in 1:50
      s=lines[rand(1:l)]
      write(f, s*"\n")
  end
  close(f)
end



function test_scope()
  # show global scope; Meta.@lower
  # varinfo()
  #=  x = 0
    for i in 1:10
      x+= i
    end
  println("x=", x)
  =#
  #varinfo()
  #return x
  #=
  n=4
  while n > 0
  println("n=", n)
  n -=1
  end 
  =#
end



function vocab(merge1)
  V = Dict("zz"=>1); empty!(V)
  for k in keys(merge1)
    vl=k[1]
    V[vl] = get(V,vl,0) + 1  

    vr=k[2]
    V[vr] = get(V,vr,0) + 1  
  end
  return V
end  



function read_dlm(file)
# skip 1-st line
  a = readdlm(file; header=true) #,skipstart=1)
  data_cells = a[1]
  header_cells = a[2] # it`s Matrix, not Array   
  #header_cells = a[2][1 , :] # it`s Matrix, not Array 
  return data_cells, header_cells
end



function plot_dlm(file, columns)
  # using DelimitedFiles, Plots
  # usage plot_dlm("filename", [1,2,4,7] ) ->column numbers

    matrix = readdlm(file; header=true)
    data_cells   = matrix[1]   
    header_cells = matrix[2]

    x=data_cells[: , 1]        # x-Axis -> first column 
    y=data_cells[: , columns]  # y-Axis -> one or more columns

    header = header_cells[1,columns] 

    header = reshape(header, 1, length(columns)) # convert vec to row

    display(plot(x,y, xlabel="blocks (1 block ~ 1.6 MB)", label=header, layout = ( length(columns),1)) )
    sleep(1);
    savefig(file * ".png")
    #savefig(file * ".pdf")

end



function poll_file_u(file)
  # (changed,renamed,timedout)
  while true
    poll_file(file,2,-1) # every 2 sec
  end
end



function subst_abc(file)
  subst = Dict("a"=>"b"); empty!(subst)
  pairs = readlines("abc-substitutions.txt")
  for i in 1:length(pairs)
    (a,b) = split(chomp(pairs[i]),'\t')
    subst[(a)] = b
  end
  text = readstring(file) ## deprecated
  chars = collect(graphemes(text))
  f = open(joinpath(pwd(),basename(file)*"_subst"),"a+")
  for i in 1:length(chars)
    out = get(subst, chars[i],chars[i])
    write(f, out)
  end
  close(f)
end



function gen_text(chars_max, word_len_min, word_len_max, sent_len_min, sent_len_max)
  abc = ('а':'я')
  signs = [".","!","?"]
  sent_len_median = (sent_len_max - sent_len_min)/2
  word_len_median = (word_len_max - word_len_min)/2
  chars_count = 0
  sentence_string = ""
  f = open("a_ru_"*string(Int(chars_max/1_000))*"K.utf8_rand","a+")
  while chars_count < chars_max
    sent_len = randn()*5  + sent_len_median
    sent_len = clamp(sent_len, sent_len_min, sent_len_max)
    sentence = [""]
    while length(sentence) < sent_len
      word_len = randn()*4  + word_len_median
      word_len = clamp(word_len, word_len_min, word_len_max)
      word = ""
      chars_count = chars_count + word_len + 1
             while length(word) < word_len
               n = rand(1:32)
               word = word * abc[n]
             end
      push!(sentence, word * " ")
    end
    chars_count = chars_count + 1
    sentence_string = uppercase(abc[rand(1:32)])
    for i in 1:length(sentence)
     sentence_string = sentence_string * sentence[i]
    end
    sentence_string = chop(sentence_string) * signs[1] * " "
    write(f, sentence_string)
    chars_count = chars_count + 1
  end
  close(f)
end



function check_config()
    #pkgs = Dict("Package"=>"version"); empty!(pkgs)
    pkgs_current = Pkg.installed()
    save_dict1_u(pkgs, "pkgs_installed.txt")
    pkgs_required = load_dict1_u("pkgs_installed.txt")
    if isequal(pkgs, pkgs_required)
        println("Packages ok")
    else println("Packages inconsistentcy!")
    end
    #=
      "CSV"               => v"0.1.2"
      "DataStreams"       => v"0.1.2"
      "SortingAlgorithms" => v"0.1.0"
      "Juno"              => v"0.2.5"
      "Conda"             => v"0.7.0"
      "SHA"               => v"0.3.3"
      "DecFP"             => v"0.1.5"
      "Hiccup"            => v"0.1.1"
      "WeakRefStrings"    => v"0.2.0"
      "Media"             => v"0.2.4"
      "PyCall"            => v"1.14.0"
      "JSON"              => v"0.13.0"
      "NullableArrays"    => v"0.0.10"
      "DataArrays"        => v"0.3.11"
      "Compat"            => v"0.31.0"
      "StatsBase"         => v"0.12.0"
      "CategoricalArrays" => v"0.1.0"
      "GZip"              => v"0.2.20"
      "MbedTLS"           => v"0.4.2"
      "MacroTools"        => v"0.3.7"
      "FileIO"            => v"0.2.1"
      "Dates"             => v"0.4.4"
      "HttpServer"        => v"0.1.7"
      "BinDeps"           => v"0.6.0"
      "ODBC"              => v"0.5.1"
      "HttpParser"        => v"0.2.0"
      "HttpCommon"        => v"0.2.6"
      "DataFrames"        => v"0.8.5"
      "Reexport"          => v"0.0.3"
      "URIParser"         => v"0.2.0"
    =#
end



function train_folder(kind,pass)
  # Train by WORDS or by SEMS
  t0=time()
  println("starting train_folder by ",kind)
  caller = "train"
  dict_vocab  = Dict( (String("word") )=>123); empty!(dict_vocab)
  dict_fc     = Dict( (String("word"),String("category") )=>123);empty!(dict_fc)
  dict_fc_cat = Dict( (String("word"),String("category") )=>123);empty!(dict_fc_cat)
  dict_param  = Dict{}("string" => 123); empty!(dict_param)
  docs_count = length(readdir(path_train))
  dict_fnames = Dict()

      for f in readdir(path_train)
          cat = basename(f)[1:CAT_NAME_LENGTH]
          dict_fnames[cat] = get!(dict_fnames,cat,0) + 1
      end
      for cat in categories
          push!(dict_param,("docs_in_cat_"*cat => get!(dict_fnames,cat,0)))
      end
  push!(dict_param,("cats_count" => length(categories)))
  #rows = 0; columns = get!(dict_param,"cats_count",0)
  dwcat = Dict("word"=>collect(1:length(categories))); empty!(dwcat)
  vwords = ["word"]; empty!(vwords)
  words_all = 0; articles_count = 0
  for cat in categories
  words_in_cat = 0; unique_words_in_cat = 0; empty!(dict_fc_cat)
  #println("cat=",cat)
    for file in readdir(path_train)
          if basename(file)[1:CAT_NAME_LENGTH] == cat
          #if contains(basename(file),cat)
              fid_in = path_train * path_div * file
              dict_fc, dict_fc_cat, words_in_file,
              dict_vocab, vwords, dwcat =
              train_doc(fid_in, cat, dict_fc, dict_fc_cat,
                        dict_vocab, vwords, dwcat)
              words_in_cat = words_in_cat + words_in_file
                    articles_count += 1;
                    if articles_count%200 == 0
                    println(articles_count)
                    end
          end
    end
    flush(STDOUT)
    unique_words_in_cat = length(dict_fc_cat)
    push!(dict_param,("unique_words_in_cat_"*cat => unique_words_in_cat))
    push!(dict_param,("words_in_cat_"*cat => words_in_cat))
    words_all = words_all + words_in_cat
  end
  #save_dict1(dwcat,"dwcat-" * kind)
  writecsv("dwcat-" * kind, dwcat)
  #mfc = hcat(vwords,mfc)
  #names = transpose(unshift!(categories,"^^^"))
  #mfc = vcat(names,mfc) #println("mfc=",mfc)
  #println("updating param-",kind)
  push!(dict_param,("unique_words_all" => length(dict_vocab)))
  push!(dict_param,("words_all" => words_all))
  push!(dict_param,("docs_all" => docs_count))

  #for cat in categories
  #delete!(dict_fc,("в",cat))
  #delete!(dict_fc,("и",cat))
  #delete!(dict_fc,("на",cat))
  #end

  #println("saving fc-",kind)
  arr_sorted = sort(collect(dict_fc), by=x->x[2], rev=true)
      open(path_work * "fc-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1][1],"\t",arr_sorted[i][1][2],"\t",arr_sorted[i][2])
        end
      end


  #println("saving param-",kind)
  arr_sorted = sort(collect(dict_param))
  #arr_sorted = sort(collect(dict_param), by=x->x[2])
      open(path_work * "param-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1],"\t",arr_sorted[i][2])
        end
      end
  t2=time()
  println("calc_mi ...")
  dict_chi, dict_mi = calc_mi()
  t3=time();
  mi_time = round((t3-t2),2)
  println("calc_mi sec=",mi_time)
  #------------------
  #prune_features(prune_limit)
  #------------------

  arr_sorted = sort(collect(dict_vocab), by=x->x[2])
  vocab_freqs = Float64[]; sizehint!(vocab_freqs,50000)
  #arr_sorted = sort(collect(dict_param), by=x->x[2])
      open(path_work * "vocab-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1],"\t",arr_sorted[i][2])
          push!(vocab_freqs,arr_sorted[i][2])
        end
      end
      max_freq = maximum(vocab_freqs)
      for i in 1:length(vocab_freqs)
      vocab_freqs[i] = vocab_freqs[i]/max_freq
      end

  stdev = std(vocab_freqs,mean=maximum(vocab_freqs))
  println("mean=",round(mean(vocab_freqs),6))

  #fid_out = path_work * "vocab-" * kind
  #save_dict1_u(dict_vocab, fid_out)

  #fid_in_sort = path_work * "vocab-" * kind
  #fid_out_sort = path_work * "vocab-" * kind * "-sort-w"
  #sort_dict1(fid_in_sort, fid_out_sort,"sort_words")

  println("categories=",length(categories), "\ntrained on ", articles_count, " docs",
  "  docs_in_dir=", length(readdir(path_train)), # docs_count,
  "  ", kind * "_all=", words_all)
  println("finish train_folder")
  t1=time()
  folder_time = round((t1-t0),2)
  #println("train folder sec=",folder_time)
  return dict_chi, dict_mi, stdev
end



function train_doc(fid_in, category, dict_fc, dict_fc_cat,
                   dict_vocab, vwords, dwcat)
  # Count words frequencies (fid_in); training by WORDS or by SEMS
  # add to vocab(dict_vocab); add to freqs; calc Mutual Information
  #t0=time()
  data = readstring(fid_in)
  words = ["word"]; empty!(words)
  if kind == "words"
  words = split(data)
  words = map(lowercase,words);
  words = clean_words(words)
  else sems = text_to_sems(data) # kind == "-sems"
  #  println(sems)
        if length(sems) != 0
            for i in 1:length(sems)
            push!(words,string(sems[i]))
            end
        end
  end

  col_num = findfirst(categories, category)#; println(c)
  w_checked = Dict("word"=>"yes"); empty!(w_checked)
  for i = 1:length(words)
      dict_vocab[words[i]] = get(dict_vocab, words[i],0) + 1
      check_word_value = get(dict_fc, (words[i],category),"new_word")

      dict_fc[(words[i],category)] = get(dict_fc,(words[i],category),0) + 1
      dict_fc_cat[(words[i],category)] = get(dict_fc_cat,(words[i],category),0) + 1
      current_word = words[i]
  dwcat = update_dwcat!(dwcat, dict_vocab, col_num, current_word, w_checked)
  end
  word_count = length(words)
  return dict_fc, dict_fc_cat, word_count, dict_vocab,
        vwords, dwcat
end



function classify_folder(kind,pass)
  # Classification by WORDS or SEMS
  #const WEB_FID_IN = "web456"
  println("starting classify_folder by ",kind)
  caller = "classify"
  if kind == "words"
  dict_fc    = load_dict21_u(path_work * "fc-words")
  dict_param = load_dict1_u(path_work * "param-words")
  elseif kind == "sems"
  dict_fc     = load_dict21_u(path_work * "fc-sems")
  dict_param  = load_dict1_u(path_work * "param-sems")
  end
  dict_class_res  = Dict()
  mconf = zeros(Int64,length(categories),length(categories));
  prob = ""; prob_summary = ""; #fid_in = ""
  debug_summary =""; count_summary = ""; articles_count = 0
  if length(readdir(path_test)) == 0
    println("Test dir is empty!")
  else
      for f in readdir(path_test)
        #if length(basename(f)) != CAT_NAME_LENGTH
        #println("filename ",f, "!= CAT_NAME_LENGTH skipping")
        #continue
        #end
      fid_in = path_test * f
      text = readstring(fid_in)
      prob,prob_summary,count_summary,debug_summary,
      sems_text,d_f1,verdict, mconf =
      classify_doc(text, kind, mconf, fid_in,
                  dict_class_res, dict_fc, dict_param)
      merge!(dict_class_res, d_f1)
            articles_count += 1;
            if articles_count%100 == 0
            #println(articles_count)
            end
      end
  end
      writecsv("mconf-" * kind, mconf)
      #println("calc_f1 ...")
      f1micr, f1macr, sumconf = calc_f1(kind)

  save_dict_ss_ss(dict_class_res,"class-res-" * kind)
  println("micro ",kind,"=",f1micr," macro ",kind,"=",f1macr) #," docs=",sumconf)
  dict_crossv[string(pass)] = (f1micr, f1macr,length(dict_fc))
  #println("words=",words)
  #println("prob=",prob)
  #println("prob_summary=",prob_summary)
  #println("verdict=",verdict)
  #println("count_summary=",count_summary)
  #println("debug_summary=",debug_summary)
  #println("categories=",length(categories), "\nclassified ", articles_count, " docs",
  #"  docs_in_dir=",length(readdir(path_test))) #,
  # "  ", kind * "_all=", words_all)

  #println("docs=",articles_count)
  println("finish classify_folder, docs=",articles_count)
  return dict_crossv
end



function classify_doc(text, kind, mconf, fid_in,
                      dict_class_res, dict_fc, dict_param)
  # Classify one doc\text by WORDS or SEMS
    debug_summary = ""; verdict = "";   #println(x)
    word_count = 0 #length(split(text)) ; i=1 ;

      unique_words_all = get(dict_param,"unique_words_all", 0)
              docs_all = get(dict_param,"docs_all", 0)

      #unique_words_all = dict_param["unique_words_all"]
      #        docs_all = dict_param["docs_all"]

    #data = readstring(fid_in)
    dict_vocab = Dict(); vocab=[]
    words = ["word"]; empty!(words)
    sems_text = ""
    if kind == "words"
    words = split(text)
    words = map(lowercase,words);
    words = clean_words(words)
    #println("words1=",words)
    elseif kind == "sems" # || caller == "web"
          sems = text_to_sems(text)
          if length(sems) != 0
              for i in 1:length(sems)
                  push!(words,string(sems[i]))
              end
              if caller == "web"
              for i in 1:min(length(sems),50)
                  sems_text = sems_text * "<br>" *
                              string(sems[i]) * "->" * view_sem(sems[i])
              end
              end
          end
    end

    if length(words) != 0
        for i in 1:length(words)
          dict_vocab[words[i]] = 1 + get(dict_vocab,words[i],0)
        end
        vocab = sort(collect(dict_vocab), by=x->x[2],rev=true)
    end
  prob = [];
    for cat in categories
      prob_cat, debug_cat, word_count =
      classify_doc_cat(words, cat, kind, dict_fc, dict_param)
      push!(prob,prob_cat); #println("prob_cat=",cat,kind,prob_cat )
      debug_summary = debug_summary * "<br>" * "\n" * debug_cat
    end
    prob_summary = ""
  #println("prob=", prob)
    prob_positive = []
    for i in 1:length(prob)
          positive_value = prob[i] - minimum(prob) + 1;
          push!(prob_positive, positive_value)
    end

    for i in 1:length(prob_positive)
      if i == findmax(prob_positive)[2]
          probes = string("\nprobability <b>",categories[i], " ",round(prob_positive[i]/sum(prob_positive),3),"<br></b>")
      else probes = string("\nprobability ",categories[i], " ",round(prob_positive[i]/sum(prob_positive),3),"<br>")
      end
      prob_summary = prob_summary * probes
    end

    cat_found_index = findmax(prob)[2]; #println("cat_found_index=",cat_found_index)
    cat_found = categories[cat_found_index];
    #println("prob=",string(prob))
    dict_class_res[(basename(fid_in),kind)] =
            (basename(fid_in)[1:CAT_NAME_LENGTH], cat_found) # * " | " * string(prob))
            # * " " *
            #string(words[1:min(length(words),3)]) *
            #"..." * prob_summary * " " * "\n")
  if fid_in != WEB_FID_IN
    row = cat_found_index
    col = findfirst(categories,basename(fid_in)[1:CAT_NAME_LENGTH])
    if col == 0 println("unknown fid_in ",basename(fid_in)[1:CAT_NAME_LENGTH])
    println("probably ",fid_in," absents in train set")
    end
    mconf[row,col] = mconf[row,col] + 1
  end

    word_count_summary = string("words in text = ",word_count)
    snippet_text = words[1:min(length(words),10)]
    snippet_vocab = ""
    if length(vocab)>0
          for i in 1:min(length(vocab),30)
            if kind == "sems" && caller == "web"
                  snippet_vocab = snippet_vocab * "<br>" *
                  string(vocab[i][1]) * "->" * string(vocab[i][2]) *
                    "->" * view_sem(parse(Int,vocab[i][1]))
            else  snippet_vocab = snippet_vocab * "<br>" *
                  string(vocab[i][1]) * "->" * string(vocab[i][2])
            end
          end
    end
  sems_text=""

  #println("snippet_vocab",snippet_vocab)
    debug_summary = debug_summary * "<br>" *
                    "unique_words_all=" * string(unique_words_all) * "<br>" *
                    "docs_all=" * string(docs_all) * "<br>" *
                    string(reshape(snippet_text,1,length(snippet_text))) * "<br>" *
                    snippet_vocab
    return prob, prob_summary, word_count_summary, debug_summary,
          sems_text, dict_class_res, verdict, mconf
end



function classify_doc_cat(words, cat, kind,
                          dict_fc, dict_param)
  #classify by WORDS
  #words = split(text)
  words = map(lowercase,words)
  #words = clean_words(words)

  docs_all = get(dict_param,"docs_all", 0)
  docs_in_cat = get(dict_param,"docs_in_cat_" * cat, 0)
  words_in_cat = get(dict_param,"words_in_cat_" * cat, 0)
  unique_words_all = get(dict_param,"unique_words_all", 0)
  prior = docs_in_cat/docs_all

  category = cat
  nbsp = "&nbsp;"
  sum_freq = 0 ; found_count =0; word_count =0;
  #println("words3="," ",category,words[1:min(length(words),3)])
  if length(words) != 0
      for i = 1:length(words)
  #        key = (words[i],category) ;
          word_count += 1
          check_word_freq = get(dict_fc, (words[i],category),"new_word")
          if check_word_freq == "new_word"
                freq = 1
          else  freq = check_word_freq + 1
                found_count =  found_count + 1
          end
          sum_freq = sum_freq + freq
      end
      prob_cat = log2(prior) + log2(sum_freq/(words_in_cat + unique_words_all))
      #prob_cat = log2(prior) + sum_freq/(words_in_cat + unique_words_all)
  else
    prob_cat = 0
  end
  #println("prob_cat=",cat, " ", prob_cat)
  debug_cat = "words_in_cat <b>" * category * "</b>" * nbsp * string(words_in_cat) *
              " sum_freq=" * string(sum_freq) *
              " found_count=" * string(found_count) *
              " prior=" * string(round(prior,2))
  return prob_cat, debug_cat, word_count
end



function text_to_sems(input)
  sentence = input #Golem2 line 1171
  golem_id, sentence, sentence_sems = sem_sentence(sentence,merge2tags,merge2sems,
                                          golem_param,sem_weight,stem_count)
  return  sentence_sems #sentence
end



function norm_mi()
end



function prune_features(kind,limit,dict_chi, dict_mi;method="mi")
 # mi=mutual information; chi=chi-square
  dict_fc = load_dict21_u(path_work * "fc-" * kind)
  #println("length(dict_fc)=",length(dict_fc))
  if   method == "mi"
      dict_scores = dict_mi
  else dict_scores = dict_chi
  end

  #println("length(dict_scores)=",length(dict_scores))
  #scores=collect(values(dict_scores));
  #p=collect(0 : 0.1 : 1) #log10(
  #med=median(scores); mid=middle(scores); mean_score=mean(scores)
  #std_score=std(scores),mean=0.00372;#co=cov(scores); cr=cor(scores)
  #q = quantile(scores,p)

  println("limit = ",limit,"  ",method," scores=",length(dict_scores)," fc=",length(dict_fc))
  println("distinct scores=",length(unique(collect(values(dict_scores)))))
  cnt = 0;
  for kk in keys(dict_scores)
  kks = (string(kk[1]),kk[2])
  if haskey(dict_fc,kks)
  val = get(dict_scores,kk,1.001)
        if val < limit
          delete!(dict_fc,(kks[1],kks[2]))
          cnt +=1
        end
  end
  end
  println("removed=",cnt,"  remained=",length(dict_fc))
  #save_dict21(dict_fc,path_work * "fc-words-")
  arr_sorted = sort(collect(dict_fc), by=x->x[2], rev=true)
      open(path_work * "fc-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1][1],"\t",arr_sorted[i][1][2],"\t",arr_sorted[i][2])
        end
      end

  return #cnt #, dict_fc
end
#limit=15;c,fc=prune_features(limit)



function calc_f1(kind)
  mconf = readcsv(path_work * "mconf-" * kind,Int64)
    p=[];r=[]; sumconf=sum(mconf)
    for row in 1:size(mconf,1)
      val= mconf[row,row]/sum(mconf[row,1:end]); if isnan(val) val=0 end;push!(p,val)
    end
    for col in 1:size(mconf,2)
      val= mconf[col,col]/sum(mconf[1:end,col]); if isnan(val) val=0 end;push!(r,val)
    end
    pmacr = mean(p);rmacr = mean(r)
    f1macr = 2 * pmacr * rmacr / (pmacr + rmacr); if isnan(f1macr) f1macr=0 end
    f1macr = round(f1macr,3)
    f1micr = []
    for i in 1:size(mconf,1)
      val = 2 * p[i] * r[i] / (p[i] + r[i]); if isnan(val) val=0 end
      push!(f1micr, val)
    end
    f1micr = round(mean(f1micr),3);
  #return ("macr=",round(f1macr,3), "micr=",round(f1micr,3),sumconf) #f1micr, f1macr
  return f1micr, f1macr, sumconf
end



function draw_table(kind)
  # draw one table in one cell: <td>TABLE</td>
  mconf = readcsv(path_work * "mconf-" * kind,Int64)
  tr=""; td=""; content=""; width=500
  td_nocolor = """<td>"""
  td_color = """<td bgcolor="#BBFFBB">"""

  names_col = append!([],categories)
  mconf = hcat(names_col,mconf)
  names_row = insert!(names_col,1,"")
  names_row = reshape(names_row,1,length(categories)+1)
  mconf = vcat(names_row,mconf)

      for row in 1:size(mconf,1)
        td=""
              for col in 1:size(mconf,2)
                td_open = td_nocolor
                if row == col && row != 1
                td_open = td_color
                end
                td = td * td_open * string(mconf[row,col]) * "</td>"
              end
        tr = "<tr>" * td * "</tr>" * "\n"
        content = content * tr
      end
      if size(mconf,2) > 10 width = width + 10*size(mconf,2) end
      table_head = """<table width=""" * string(width) * """ cellpadding="5">"""
      html_table = "<td>" * table_head * content * "</table></td>"
      println(html_table)
  return html_table
end



  #kind="words";h=draw_table(kind)
  #=
  <table width="100%">
  <tr>
  -----
  TABLE1
  -----
  TABLE2
  -----
  </tr>
  </table>
  =#



function update_dwcat!(dwcat, dict_vocab, col_num,
                     current_word, w_checked)
  if get(w_checked, current_word,"new") == "yes"
  else
          w_checked[current_word] = "yes"
        if  haskey(dwcat, current_word)
          arr = get(dwcat, current_word,[9])
          arr[col_num] = arr[col_num] + 1
          dwcat[current_word] = arr
        else
          arr = zeros(Int,length(categories))
          arr[col_num] = arr[col_num] + 1
          dwcat[current_word] = arr
        end
  end
  return dwcat
end



function load_dwcat()
  t2=time()
  dwcat = readcsv(path_work * "dwcat-" * kind) #,String)
  words = dwcat[:,1]
  mfc  = dwcat[:, 1:size(dwcat,2) .!=1]
  for i in 1:size(mfc,1)
  mfc[i,1] = parse(Int,replace(mfc[i,1],"[",""))
  mfc[i,size(mfc,2)] = parse(Int,replace(mfc[i,size(mfc,2)],"]",""))
  end
  t3=time();
  load_dwcat_time = round((t3-t2),2)
  #println("load_dwcat size=",size(mfc)," sec=",load_dwcat_time)
  return words, mfc
end



function calc_mi()
  #calculate Mutual Information, save in dict_ss_1111
  #ReuterCV
  #n_all=801948;n11=49;n01=141;n10=27652;n00=774106 #mi=0.0001105 =73.458
  words,mfc = load_dwcat()
  chi = 0
  learned_param = load_dict1_u("param-" * kind)
  n_all = get(learned_param,"docs_all",999999)
  #dict_mi_nnnn = Dict();
  dict_mi = Dict(); dict_chi = Dict()
  n11=0;n01=0;n10=0;n00=0; smoo=0.001
  cat   = ""; words_count=0
  #println(mfc);
  #println("sum=",sum(mfc));println("countnz=",countnz(mfc))
  for w in 1:length(words)
  cat_num = 0;
  words_count +=1
  if (words_count)%(round(length(words)/10)) == 0
    #println(round(words_count*100/length(words))," %")
    flush(STDOUT)
  end
  for cat in categories
    kk = (words[w], cat); cat_num +=1
    #println("kk=",kk)
    row = w
    col = findfirst(categories,cat)
  #println("",round(row/length(words),2))
    n11 = mfc[row,col];
    #ndocs = docs_in_cat[cat_num]
  ndocs = get!(learned_param,"docs_in_cat_"*cat,0)
      if true #n11 != 0
        n01 = ndocs - n11
        n10 = sum(mfc[row,1:end]) - n11
        n00 = n_all - (n11 + n01 + n10)
        n11 = n11 + smoo;  n01 = n01 + smoo
        n10 = n10 + smoo;  n00 = n00 + smoo
        #println("nnnn=",(n11,n01,n10,n00))
        #dict_mi_nnnn[kk] = (n11,n01,n10,n00)
        mi = (n11/n_all)*log2(  (n_all*n11)/( (n10+n11)*(n01+n11) )  ) +
              (n01/n_all)*log2(  (n_all*n01)/( (n00+n01)*(n01+n11) )  ) +
              (n10/n_all)*log2(  (n_all*n10)/( (n10+n11)*(n00+n10) )  ) +
              (n00/n_all)*log2(  (n_all*n00)/( (n00+n01)*(n00+n10) )  )
        #n11=49;n01=141;n10=27652;n00=774106; n_all=801948

        #n11=6.6;n01=183.4;n10=27694.4;n00=774063.6
        chi = ( (n11+n10+n01+n00)*(n11*n00-n10*n01)^2 )/
        ( (n11+n01)*(n11+n10)*(n10+n00)*(n01+n00) )
          # println("kk=",kk)
          dict_mi[kk] = round(mi,4)
          dict_chi[kk] = round(chi,1)
          # println("dict_mi[kk] =",dict_mi[kk])
              #if  isnan(dict_mi[kk])
                #   pop!(dict_mi,kk,"hz")
              #end
      end
  end
  end
  for cat in categories # create sorted dict_mi_* for every cat
    dict_mi_cat = Dict()
    for kk in keys(dict_mi)
      if kk[2] == cat
      key = kk;  val = dict_mi[key]
      dict_mi_cat[key] = val
      end
    end

  arr_sorted = sort(collect(dict_mi_cat), by=x->x[2], rev=true)
      open("mi-" * cat * "-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1][1],"\t",arr_sorted[i][1][2],"\t",arr_sorted[i][2])
        end
      end
  end
  #println("writecsv() sorted mi-" * kind)

  arr_sorted = sort(collect(dict_mi), by=x->x[2], rev=true)
      open(path_work * "mi-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1][1],"\t",arr_sorted[i][1][2],"\t",arr_sorted[i][2])
        end
      end
  arr_sorted = sort(collect(dict_chi), by=x->x[2], rev=true)
      open(path_work * "chi-" * kind, "w") do f
        for i in 1:length(arr_sorted)
          println(f,arr_sorted[i][1][1],"\t",arr_sorted[i][1][2],"\t",arr_sorted[i][2])
        end
      end

  return dict_chi, dict_mi
end



  #calc_mi()
  #=dict_mi_nnnn, dict_mi = calc_mi(dict_param)
  datafile = "C:\\factbook\\GOLEM_2017\\golem-1.4.4\\fc-word-docs"
  fid_out = path_work * "mi-nnnn"
  save_dict1(dict_mi_nnnn, fid_out)
  fid_out = path_work * "mi"
  save_dict1(dict_mi, fid_out)
  =#



function deldirs(path_work,dirs)
  for dir in dirs
    rm(path_work * dir,force=true, recursive=true)
  end
  return 0
end



function mkdirs(path_work, dirs)
  # dirs = ["string1","string2", ...]
  for dir in dirs
    if !isdir(path_work * dir)
        mkdir(path_work * dir)
    end
  end
  println("created ", length(dirs), " directories")
  return 0
end



function clean_words(words)
  clean_words = String[]; s=""
  if length(words) > 0
    for i in 1:length(words)
      if (m = match(r"""^(\p{L}\.)+$|\p{L}{1,}(\-\p{L}+)?(\-\p{N}+)?|[\-\p{Sc}]?
          [\p{N}\-:,./()]*\p{N}[%°']?""", words[i])) !== nothing
          s = string(m.match)
      end
    push!(clean_words, s)
    end
  end
  #ends = [",",".",":","!","?",";",")"]; starts = [",",".",":","!","?",";","("]
  #if endswith(words[i], ",") words[i] = chop(words[i])   end
  return clean_words
end



  #words=["a!","b.","bb.","c,","d?","e:","f;","g)","(h"]; words=""
  #clean=clean_words(words)



function load_stops()
  stops = load_dict1_u("stops")
  println("loading ",length(stops)," stop-words")
  return stops
end



function load_markup()
  lines = readlines("tegi_zakupki.txt")
  for i in 1:length(lines)
  fname = split(lines[i],"###")[2][1:15]
   write("docs-gz/"*string(fname)*string(i),lines[i])
 end

end



function parse_doc(doc, categories, kind, find_cat, min_chars)
  # parse and classify paragraphs
  #fid_in  = open(input_file)
  # parse_doc("перевод текста", categories, "words","perev", "200")
  
  #input_file = path_work * "test-big"; cat = "gar" ;
  #categories=["adr","fin","gar","req","tec"]
  #min_chars = 50
  #d,f,para1=parse_doc(input_file, categories, cat, min_chars)
  caller = "web"
  dict_para = Dict(); f = Dict()#((123,123) => (1.0,1.0,1.0,1.0)); pop!(dict_para,(123,123))
  para_count=0; prob=[]; cat=find_cat; w1=0; w2=0; w3=0; hz="нет категории"
  para_num1=0; para_num2=0; para_num3=0; para1=""; para2=""; para3=""
  min_chars=parse(Int,min_chars); para_analyzed=0; para_limit_exceeded = ""
  pc=0; pa=0
  words_count=0; wc=0; p1=""; chars=0
  lines = split(doc,'\n');
  println("length(lines)=",length(lines))
  #if length(readlines(input_file))<=5000
  if length(lines)<=500 #&& length(lines) > 0
  #while !eof(fid_in) #length(dump)
  for i in 1:length(lines)
      #line = readline(fid_in)
      line = lines[i]
      para_count += 1
            if startswith(line,"#") || length(line) <= min_chars # (CR LF) = > 2 chars
              continue
            end
      para_analyzed +=1
      words = line; l = length(split(line))# print("words=",words)
      words_count = words_count + l; chars = chars + length(words)
      text = words

  fid_in = WEB_FID_IN
  #kind = "words"
  if kind == "words"
  dict_fc    = load_dict21_u(path_work * "fc-words")
  dict_param = load_dict1_u(path_work * "param-words")
  elseif kind == "sems"
  dict_fc     = load_dict21_u(path_work * "fc-sems")
  dict_param  = load_dict1_u(path_work * "param-sems")
  end
  dict_class_res  = Dict()
  mconf = zeros(Int64,length(categories),length(categories))

      #fid_in = "WEB"

      prob,prob_summary,count_summary,debug_summary,
      sems_text,d_f1,verdict, mconf =

      classify_doc(text, kind, mconf, fid_in,
                  dict_class_res, dict_fc, dict_param)

      key = (maximum(prob)/sum(prob), categories[findmax(prob)[2]], para_count)
      dict_para[key] = l
      f=filter((k, v) -> k[2] == cat, dict_para)
      if !isempty(f) para_num1 = maximum(f)[1][3]; w1 = maximum(f)[1][1]; pop!(f,maximum(f)[1])
      if !isempty(f) para_num2 = maximum(f)[1][3]; w2 = maximum(f)[1][1]; pop!(f,maximum(f)[1]) end
      if !isempty(f) para_num3 = maximum(f)[1][3]; w3 = maximum(f)[1][1]; pop!(f,maximum(f)[1]) end
      else para_num1 = 0; para_num2 = 0; para_num3 = 0;
      end
      #println("prob=",prob)
      #println("prob_summary=",prob_summary); println("debug_summary=",debug_summary)
  end
  #para1 = para_num1==0 ? hz: readlines(input_file)[para_num1]
  #para2 = para_num2==0 ? hz: readlines(input_file)[para_num2]
  #para3 = para_num3==0 ? hz: readlines(input_file)[para_num3]
  #close(fid_in)
  para1 = para_num1==0 ? hz : lines[para_num1]; w1 = round(w1,3)
  para2 = para_num2==0 ? hz : lines[para_num2]; w2 = round(w2,3)
  para3 = para_num3==0 ? hz : lines[para_num3]; w3 = round(w3,3)
  println("\ncat=",cat," weight=",round(w1,3),"\n",para_num1==0 ? hz : para1)
  println("\ncat=",cat," weight=",round(w2,3),"\n",para_num2==0 ? hz : para2)
  println("\ncat=",cat," weight=",round(w3,3),"\n",para_num3==0 ? hz : para3)
  println("\nparagraphs in doc=",para_count);
    pc = para_count
  println("\nparagraphs analyzed=",para_analyzed)
    pa = para_analyzed
  #else para_limit_exceeded = string(length(readlines(input_file))) *
  p1 = lines[1]; wc = words_count

  else para_limit_exceeded = string(length(lines)) *
  " абзацев - слишком много для теста"
    println(para_limit_exceeded)
  end
  return  p1,para1,para2,para3,w1,w2,w3,pc,pa,wc,chars,para_limit_exceeded
end



function sum_dict_ss_1(dict1,dict2)
  # merge dicts (a)->1 or (a,b)->1, summarize values if same key; commutative
  kk = keys(dict2)
  for k in kk
    dict1[k] = get(dict1,k,0) + dict2[k]
  end
  return dict1
end



  #a = Dict(("a","b")=>1,("a","c")=>1); b = Dict(("a","b")=>2,("ca","c")=>1)
  #c=sum_dict_ss_1(a,b)



function sum_dict_ss_11(dict1,dict2)
  # merge dicts (a)->1 or (a,b)->1, summarize values if same key; commutative
    for kv in dict2           # kv ->   (key,value)
      if !haskey(dict1,kv[1]) # kv[1] -> key of kv
          push!(dict1,kv)
      else sum_1 = get(dict1,kv[1],"no_key")[1] + get(dict2,kv[1],"no_key")[1]
          sum_2 = get(dict1,kv[1],"no_key")[2] + get(dict2,kv[1],"no_key")[2]
          val = (sum_1,sum_2)
          push!(dict1,(kv[1] => val))
      end
    end
  return dict1
end
  #a = Dict(("a","b","c")=>(1),("a","c","d")=>(3)); b = Dict(("a","b","f")=>(22),("c","a","c")=>(44))
  #c=sum_dict_ss_11(a,b)
  #c=append_dict!(a,b)



function sum_dict_ss_1111(dict1,dict2)
  # merge dicts (a)->1 or (a,b)->1, summarize values if same key; commutative
    for kv in dict2           # kv ->   (key,value)
      if !haskey(dict1,kv[1]) # kv[1] -> key of kv
          push!(dict1,kv)
      else sum_11 = get(dict1,kv[1],"no_key")[1] + get(dict2,kv[1],"no_key")[1]
          sum_01 = get(dict1,kv[1],"no_key")[2] + get(dict2,kv[1],"no_key")[2]
          sum_10 = get(dict1,kv[1],"no_key")[3] + get(dict2,kv[1],"no_key")[3]
          sum_00 = get(dict1,kv[1],"no_key")[4] + get(dict2,kv[1],"no_key")[4]
          val = (sum_11,sum_01,sum_10,sum_00)
          push!(dict1,(kv[1] => val))
      end
    end
  return dict1
end



  #a = Dict(("a")=>(1,1,1,1),("c")=>(2,2,2,2));
  #b = Dict(("a")=>(3,4,5,6),("abcd")=>(4,6,7,8))
  #c=sum_dict_ss_1111(b,a)



function save_dict(dict, datafile)
  k="";v=""
  for k in keys(dict)  v = get(dict,k,"empty dict"); break end
      lkey = length(k); lval = length(v)
  open(datafile, "w") do f
    for kk in keys(dict)
      k_string=""; v_string = ""
      for i in 1:lkey  k_string = k_string * string(kk[i]) * "\t" end
      vv = get(dict,kk,"hz")
      for i in 1:lval  v_string = v_string * string(vv[i]) * "\t" end
      println(f,k_string,chop(v_string))
    end
  end
end



  #dict = Dict((string(i),"class") => (i) for i = 1:10)
  #file = path_work * "alldicts"
  #@time save_dict(dict, file)
  #@time save_dict21(dict, file)



function load_dict(datafile,key_length)
  # key->(a,b) or (a) or (1,a); val->(1)...(1,2,3,4)
  dict = Dict()
  fid = open( datafile )
  while !eof(fid)
      key = ""; val =""
      line = readline(fid)
      #arr = (split(chomp(line),"\t\t"))
      arr = (split(chomp(line),"\t"))
      key=()
          for i = 1:key_length
            key = (key...,arr[i])
          end
      val=()
          for i = key_length + 1 : length(arr)
            val = (val...,parse(Int,arr[i]))
          end
      dict[key]= val
  #    key_arr = split(arr[1],"\t")
  #    dict[key]= val
  end
  close(fid)
  return dict
  #println("k=",key);println("v=",val)
end



function load_dict_0(datafile)
  # key->"abc" ; val->(1)...(1,2,3,4)
  dict = Dict()
  fid = open( datafile )
  while !eof(fid)
    key = ""; val =""
    line = readline(fid)
    #arr = (split(chomp(line),"\t\t"))
    arr = (split(chomp(line),"\t"))
    key=()
    for i = 1:key_length
      key = (key...,arr[i])
    end
    val=()
    for i = key_length + 1 : length(arr)
      val = (val...,parse(Int,arr[i]))
    end
    dict[key]= val
    #    key_arr = split(arr[1],"\t")
    #    dict[key]= val
  end


  close(fid)
  return dict
  #println("k=",key);println("v=",val)
end



function save_dict1_u(dict1, datafile)
  # Dict[(k1) => Int]
  #print("Saving ", datafile, " ... ")
  arr_out = sort(collect(dict1), by=x->x[1])
  #arr_out = sort(collect(dict1), by=x->x[2])
  #arr_out = sort(collect(dict1), by=x->x[2] , rev=true)
  #=
  open(datafile, "w") do f #@time
    for k in keys(dict1)
     #println(f,k,"\t",dict1[k])
      println(f,k)
    end
  end
  =#  
  open(datafile, "w") do f
    for i in 1:length(arr_out)
      println(f,arr_out[i][1],"\t",arr_out[i][2])
    end
  end
end



function save_dict1_u_by1_desc(dict1, datafile)
  # Dict[(k1) => Int]
  arr_out = sort(collect(dict1), by=x->x[1], rev=true)
  open(datafile, "w") do f
    for i in 1:length(arr_out)
      println(f,arr_out[i][1],"\t",arr_out[i][2])
    end
  end
end



function save_dict3(dict3, output_file)
  # dict3 = Dict( (Int, Int) => "morpheme1  morpheme2 ...")
  # (m, freq)	=> "morphemes"; sort by freq; output_file="morphemes_.tsv"
  arr_out = sort(collect(dict3), by=x->x[1][2], rev=true)  # by freq desc
  # arr_out = sort(collect(dict3), by=x->x[1][1], rev=false) # by m asc
  open(output_file, "w") do f
    for i in 1:length(arr_out)
      println(f,arr_out[i][1][1],'\t',arr_out[i][1][2],'\t',arr_out[i][2])
    end
  end
end



function save_dict3_hash_freq_sent(dict3, output_file)
    # dict3 = Dict( (UInt64, Int) => "sentence")
    # (hash, freq)	=> "sentence"; sort by freq; output_file="..._hash.tsv"
    arr_out = sort(collect(dict3), by=x->x[2], rev=true)  # by freq desc
    # arr_out = sort(collect(dict3), by=x->x[1][1], rev=false) # by m asc
    open(output_file, "w") do f
      for i in 1:length(arr_out)
        println(f,arr_out[i][2],'\t',arr_out[i][1][2])
      end
    end
end



function load_dict3(input_file)
  # dict3 = Dict( (Int,Int) => "morpheme1  morpheme2 ..."); input_file="morphemes.tsv"
  # m	\t freq	\t morphemes
  dict3 = Dict()
  fid = open( input_file )
  while !eof(fid)
    line = readline(fid)
    (m, freq, morphemes) = split(chomp(line),'\t')
    m = parse(Int,m); freq = parse(Int,freq)
    dict3[(m,freq)] = morphemes
  end
  close(fid)
  return dict3    
end



function load_dict3_hash(input_file)
  # dict3 = Dict((UInt64, Int) => "sentence"); input_file="_hash.tsv"
  # (hash, freq)	\t sentence
  dict3 = Dict()
  fid = open(input_file)
  while !eof(fid)
    line = readline(fid)
    t = (ha, freq) = split(chomp(line),'\t')[1]
    h = t[1]; freq = parse(Int, t[2])
    h = parse(UInt64, h); #freq = parse(Int, freq)
    #dict3[(h, freq)] = sentence
    dict3[(h, freq)] = "sentence" #sentence
  end
  close(fid)
  return dict3    
end



function load_dict_s_s(datafile)
  # dict_s_s -> "a" => "b"
  dict_s_s = Dict("a" => "b")
  empty!(dict_s_s)
  fid = open(datafile)
  while !eof(fid)
    line = readline(fid)
    ss = split(chomp(line),'\t')
    #    println("nn[1]=",nn[1])
    dict_s_s[ss[1]] = ss[2]
  end
  close(fid)
  return dict_s_s
end



function save_dict1_u_by2_desc(dict1, datafile; lines_num=nothing)
  # Dict[(k1) => Int]
  arr_out = sort(collect(dict1), by=x->length(x[1]), rev=true)
  # filter!(arr_out, by=x[2] > 2)
  if !(lines_num === nothing) && length(dict1) > lines_num
    resize!(arr_out, lines_num) #retain top-lines_num
    println("truncated from ", length(dict1), "   to ", lines_num)
  end  
  open(datafile, "w") do f
    for i in 1:length(arr_out)
      #if !startswith(arr_out[i],"")
      if arr_out[i] != ""
      println(f,arr_out[i][1],"\t",arr_out[i][2])
      end
    end
  end
end



function save_dict1_u_by1_desc(dict1, datafile; lines_num=nothing)
    # Dict[(k1) => Int]
    filter!(p->p.second > 5, dict1)
    arr_out = sort(collect(dict1), by=x->length(x[1]), rev=true)
    arr_keys = sort(collect(keys(dict1)), by=x->length(x[1]), rev=true)
    arr_lenkeys = map(x->length(x), arr_keys)
    maxlen = maximum(arr_lenkeys)
    println("maxlen=", maxlen)
    arr_unique = []; arr_total = []; stat = [];  arr_str = [];  sum_unique = 0;  sum_total = 0
    d2 = Dict("a"=>1)
    for i in 1:maxlen
        chunk = filter(p->length(p.first) == i, dict1)
        chunk_sorted = sort(collect(chunk), by=x->x[2], rev=true)

        sumval = sum(collect(values(chunk))) ; push!(arr_total, sumval)
        len_chunk = length(chunk) ; push!(arr_unique, len_chunk)

        sum_total += sumval; sum_unique += length(chunk)
        #println("i=",i,"  length(chunk)",length(chunk)) 
        report_string = "unique " * string(i) * "-chars = " * string(length(chunk)) * 
                        "   total = " * string(sumval) 
        push!(stat, report_string)
        i +=1
        if length(chunk_sorted) > 0
            for j in 1:length(chunk_sorted)
              str = chunk_sorted[j][1] * '\t' * string(chunk_sorted[j][2]) * 
                    '\t' * string(chunk_sorted[j][2] / sum_unique)
              push!(arr_str, str)
            end
        end
        #append!(arr_str, chunk_sorted)
        merge!(d2, chunk)
    end  
    println("len d2=", length(d2))

    pushfirst!(arr_str, "--------------------------------------------")
    report_string = "sum_unique = " * string(sum_unique) * '\t' *
    "sum_total = " * string(sum_total) 
    pushfirst!(arr_str, report_string)
    pushfirst!(arr_str, "--------------------------------------------")
    
    #using Plots            
    title = "Распределение длин слов" 
    uval = map(x->length(x), collect( keys(d2) ) )
    plt = histogram(uval, bins=1:maxlen, title = "TinyStories", 
                    xlabel="длина слова", ylabel="различных слов")
    savefig(plt, "word_lengths.png")

    arr = arr_total ./ sum_total
    arr_ = log10.(arr)
    plt = plot([i for i in 1:maxlen], arr_, 
          title = "TinyStories total", label="вероятность слова данной длины", 
          xlabel="длина слова", ylabel="log10")
    savefig(plt, "word_probs.png")
    
    arr = arr_unique ./ sum_total
    arr_ = log10.(arr)
    plt = plot!([i for i in 1:maxlen], arr_, 
          title = "TinyStories", label="распределение  слов по длинам (гистограмма)", 
          xlabel="длина слова", ylabel="log10")
    savefig(plt, "word_probs.png")

    if !(lines_num === nothing) && length(dict1) > lines_num
      resize!(arr_out, lines_num) #retain top-lines_num
      println("truncated from ", length(dict1), "   to ", lines_num)
    end 

    #arr_out = new 
    prepend!(arr_str, stat)

      open(datafile, "w") do f
        for i in 1:length(arr_str)
            if arr_str[i] != ""
                println( f, arr_str[i] )
            end
        end
      end
print(arr_unique); #print(arr_total)
end


# график вероятностей морфем длины L
# N(L)/N_unique(L)/sum(N(L)). 
# Где N(L) - общее число морфем длины L, а 
# N_unique(L) - разнообразие морфем длины L
# в логарифмическом масштабе, т.к. вероятности будут маленькие


function save_dict1_u_by2_desc_io(dict1, output_file; lines_num=nothing)
  # Dict[(k1) => Int]
  io = IOBuffer()
  arr_out = sort(collect(dict1), by=x->x[2], rev=true)
  if !(lines_num === nothing) && length(dict1) > lines_num
    resize!(arr_out, lines_num) #retain top-lines_num
    println("truncated from ", length(dict1), "   to ", lines_num)
  end  
  #open(datafile, "w") do f
    for i in 1:length(arr_out)
      #if !startswith(arr_out[i],"")
      if arr_out[i] != ""
      println(io, arr_out[i][1],"\t",arr_out[i][2])
      end
    end
  #end

  tsv = String(take!(io)) 
  write(output_file, tsv) 
end



function load_dict1_u(datafile; num_type=Int)
  # Dict[(k1) => Int or Float64]
  if num_type == Int
  dict1 = Dict{String, Int64}(String("a") => 0)
  pop!(dict1,String("a"))
  else
  dict1 = Dict{String, Float64}(String("a") => 0.0)
  pop!(dict1,String("a"))
  end
    fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      #(k1,count) = split(chomp(line),'\t')
      #k1 = split(chomp(line),'\t')[1]
      #count = split(chomp(line),'\t')[2]
      #if count == "" prinln(k1) end
      #count = split(chomp(line),'\t')[2]
      (k1,count) = split(chomp(line),'\t')
      try
      if num_type == Int
        dict1[k1] = parse(Int,count)
      else
        dict1[k1] = parse(Float64,count)
      end
      catch
        continue
      end
    end
    close(fid)
    return dict1
end



function load_dict1_u_hash(datafile)
  # Dict[(k1) => Int or Float64]
  #if num_type == Int
  dict1 = Dict{UInt64, Int64}(0xb4e92987fa06fcab => 0)
  #pop!(dict1,String("a"))
  #else
  #dict1 = Dict{String, Float64}(String("a") => 0.0)
  #pop!(dict1,String("a"))
  #end
    fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (k1,count) = split(chomp(line),'\t')
      try
  #    if num_type == Int
        #dict1[k1] = parse(Int,count)
        k2 = parse(UInt64,k1)
        dict1[k2] = parse(Int,count)
  #    else
  #      dict1[k1] = parse(Float64,count)
  #    end
      catch
        continue
      end
    end
    close(fid)
    return dict1
end



function load_dict_s_arr(dict1, datafile)
  # Dict[(k1) => Int]
  #print("Saving ", datafile, " ... ")
   open(datafile, "w") do f #@time
    for k in keys(dict1)
      println(f,k,"\t",dict1[k])
    end
  end
end



function load_dict_s_arr(datafile)
  # Dict[(k1) => Int]
  dict = Dict{String, Array}(String("a") => [1,2,3])
  pop!(dict,String("a"))
  fid = open( datafile )
  while !eof(fid)
    line = readline(fid)
    (k1,arr) = split(chomp(line),'\t')
    try
      dict[k1] = arr
    catch
      continue
    end
  end
  close(fid)
  return dict
end



function sort_dict1(input_file, output_file, what_sort)
  datafile = input_file
  d_in = load_dict1_u(datafile)
      if what_sort == "sort_words"
        arr_out = sort(collect(d_in))
      else #what_sort == "sort_numbers"
        arr_out = sort(collect(d_in), by=x->x[2])
      end
  datafile = output_file
      open(datafile, "w") do f
        for i in 1:length(arr_out)
          println(f,arr_out[i][1],"\t",arr_out[i][2])
        end
      end
end



function save_dict21_u(dict_21, datafile)
  # Dict[("word","class") => 123]
  #print("Saving ", datafile, " ... ")
   open(datafile, "w") do f #@time
    for k in keys(dict_21)
      println(f,k[1],"\t",k[2],"\t",dict_21[k])
    end
  end
end



function save_dict21_u_by2_desc(dict_21, datafile)
  # Dict[("word","class") => 123]
  arr_out = sort(collect(dict_21), by=x->x[2], rev=true)
   open(datafile, "w") do f #@time
    for i in 1:length(arr_out)
      println(f,arr_out[i][1],"\t",arr_out[i][2]) #,"\t",arr_out[i][3])
    end

  #  for k in keys(dict_21)
  #    println(f,k[1],"\t",k[2],"\t",dict_21[k])
  #  end
    end
end



function load_dict21_u(datafile; num_type=Int)
  # Dict[(k1,k2) => Int]
  if num_type == Int
    dict_21 = Dict((String("a"),String("b")) => 0)
    pop!(dict_21,(String("a"),String("b")))
  else
    dict_21 = Dict((String("a"),String("b")) => 0.0)
    pop!(dict_21,(String("a"),String("b")))
  end
    #print("Reading ", datafile, " ... ")
    #time0 = time()
    fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (k1,k2,count) = split(chomp(line),'\t')
      try
      if num_type == Int
        dict_21[(k1,k2)] = parse(Int,count)
      else
        dict_21[(k1,k2)] = parse(Float64,count)
      end
      catch
        continue
      end
    end
    #dtime = time()-time0
    #println(dtime, " sec")
    close(fid)
    return dict_21
end



function sort_dict21(input_file, output_file, what_sort)
  datafile = input_file
  d_in = load_dict21(datafile)
      if what_sort == "sort_words"
        arr_out = sort(collect(d_in))
      else #what_sort == "sort_numbers"
        arr_out = sort(collect(d_in), by=x->x[2])
      end
  datafile = output_file
      open(datafile, "w") do f
        for i in 1:length(arr_out)
          println(f,arr_out[i][1],"\t",arr_out[i][2],"\t") #,arr_out[i][3])
        end
      end
end



function load_dict_1s_1(datafile)
  # Dict[(Int,k2) => Int] # dict_1s_1 -> (1,"s")=>1
  #datafile = "C:\\articles\\bayes\\tele2\\fc-sems"
  # d=load_dict_1s_1(datafile)
  dict_1s_1 = Dict((12345,String("s")) => 0)
  pop!(dict_1s_1,(12345,String("s")))
  #print("Reading ", datafile, " ... ")
  #time0 = time()
  fid = open( datafile )
  while !eof(fid)
    line = readline(fid)
    (k1,k2,count) = split(chomp(line),'\t')
    try
      dict_1s_1[(parse(Int,k1),k2)] = parse(Int,count)
    catch
      continue
    end
  end
  #dtime = time()-time0
  #println(dtime, " sec")
  close(fid)
  return dict_1s_1
end



function save_dict_1s_1(dict_1s_1, datafile)
  # dict_1s_1 -> (1,"s")=>1
  #print("Saving ", datafile, " ... ")
  open(datafile, "w") do f #@time
    for k in keys(dict_1s_1)
      println(f,k,"\t",dict_1s_1[k])
    end
  end
end



function load_dict_ss_11(datafile)
  # dict_ss_11 -> ("a","b")=>(1,1)
  dict_ss_11 = Dict((String("ab"),String("bc")) => (1,1))
  pop!(dict_ss_11,("ab","bc"))
  fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (s1,s2,n1,n2) = split(chomp(line),'\t')
  #    println("nn[1]=",nn[1])
      dict_ss_11[(s1,s2)] = (parse(Int,n1),parse(Int,n2))
    end
  close(fid)
  return dict_ss_11
end



function load_dict_ss_1(datafile)
  # dict_ss_1 -> ("a","b")=>1)
  #dict_ss_1 = Dict((String("ab"),String("bc")) => 1)
  #pop!(dict_ss_1,("ab","bc"))
  dict_ss_1=Dict()
  fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (s1,s2,n) = split(chomp(line),'\t')
  #    println("nn[1]=",nn[1])
      dict_ss_1[(s1,s2)] = parse(Float64,n)
    end
  close(fid)
  return dict_ss_1
end



function save_dict_ss_11(dict_ss_11, datafile)
  # dict_ss_11 -> ("a","b")=>(1,1)
  #dict_mi = Dict(("a","b") => (4,1))
  #datafile = "C:\\factbook\\GOLEM_2017\\golem-1.4.4\\ss11"
  #d = save_dict_ss_11(dict_mi, datafile)

  open(datafile, "w") do f #@time
    for k in keys(dict_ss_11)
      println(f,k[1],"\t",k[2],"\t",
      dict_ss_11[k][1],"\t",dict_ss_11[k][2],"\t")
    end
  end
end



function load_dict_ss_1111(datafile)
  # dict_ss_1111 -> ("a","b")=>(1,1,1,1)
  #datafile = "C:\\factbook\\GOLEM_2017\\golem-1.4.4\\ss1111"
  #d=load_dict_ss_1111(datafile)

  dict_ss_1111 = Dict((String("ab"),String("bc")) => (1,1,1,1))
  pop!(dict_ss_1111,("ab","bc"))
  fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (s1,s2,n1,n2,n3,n4) = split(chomp(line),'\t')
      dict_ss_1111[(s1,s2)] = (parse(Int,n1),parse(Int,n2),parse(Int,n3),parse(Int,n4))
    end
  close(fid)
  return dict_ss_1111
end



function save_dict_ss_1111(dict_ss_1111, datafile)
  # dict_ss_1111 -> ("a","b")=>(1,1,1,1)
  #print("Saving ", datafile, " ... ")
  #dict_mi = Dict(("a","b") => (6,2,4,1))
  #datafile = "C:\\factbook\\GOLEM_2017\\golem-1.4.4\\ss1111"
  #d = save_dict_ss_1111(dict_mi, datafile)

  open(datafile, "w") do f #@time
    for k in keys(dict_ss_1111)
      println(f,k[1],"\t",k[2],"\t",
      dict_ss_1111[k][1],"\t",dict_ss_1111[k][2],"\t",
      dict_ss_1111[k][3],"\t",dict_ss_1111[k][4]," ")
    end
  end
end



function load_dict_ss_ss(datafile)
  # dict_ss_11 -> ("a","b")=>(1,1)
  #datafile = "C:\\factbook\\GOLEM_2017\\golem-1.4.4\\fc-words-docs"
  #d=load_dict_ss_11(datafile)

  dict_ss_ss = Dict((String("a"),String("b")) => (String("c"),String("d")))
  empty!(dict_ss_ss)
  fid = open( datafile )
    while !eof(fid)
      line = readline(fid)
      (s1,s2,s3,s4) = split(chomp(line),'\t')
  #    println("nn[1]=",nn[1])
      dict_ss_ss[(s1,s2)] = (s3,s4)
    end
  close(fid)
  return dict_ss_ss
end



function save_dict_ss_ss(dict_ss_ss, datafile)
  # dict_ss_ss -> ("a","b")=>("a","b")
  open(datafile, "w") do f #@time
    for k in keys(dict_ss_ss)
      println(f, k[1],"\t", k[2],"\t",
      dict_ss_ss[k][1],"\t", dict_ss_ss[k][2])
    end
  end
end



function save_dict22(dict_22, datafile)
  # Dict[(Int1,Int2) => (Int3,Int4)]
 open(datafile, "w") do f
    for k in keys(dict_22)
      if length(k) == 2
        println(f,k[1],"\t",k[2],"\t",dict_22[k][1],"\t",dict_22[k][2])
      end
    end
  end
end



function load_dict22(datafile)
  # Dict[(Int1,Int2) => (Int3,Int4)]
  dict_22 = Dict((1,1) => (0,0))
  pop!(dict_22,(1,1))
  fid = open( datafile )
  while !eof(fid)
    line = readline(fid)
    (k1,k2,k12,count) = split(chomp(line),'\t')
    try
      dict_22[(parse(Int,k1),parse(Int,k2))] = (parse(Int,k12),parse(Int,count))
    catch
      continue
    end
  end
  close(fid)
  return dict_22
end



function load_dictionary(datafile)
  # file => Dict(Any)
  dictionary = Dict()
  fid = open( datafile )
  while !eof(fid)
    kd = deserialize(fid)
    key = kd[1]; value = kd[2]
    dictionary[key] = value
  end
  close(fid)
  return dictionary
end



function load_dictionary_u(datafile)
  # file => Dict(Any)
  dictionary = Dict()
  fid = open( datafile )
  while !eof(fid)
    kd = deserialize(fid)
    key = kd[1]; value = kd[2]
    dictionary[key] = value
  end
  close(fid)
  return dictionary
end



function save_dictionary_u(dictionary, datafile)
  # Dict(Any) => file
  #print("Saving ", datafile, " ... ")
  open(datafile, "w") do fid
    for k in keys(dictionary)
      kd = (k,dictionary[k])
      serialize(fid,kd)
    end
  end
end



function dat_tsv(datafile)
  #datafile = "phrases_sem_1.dat"
  #dat_tsv("phrases_sem_1.dat")

  phrases_sem = load_dictionary(datafile)
  name = splitext(datafile)[1]
  save_dict1_u(phrases_sem, name * ".tsv")
  println("saved ", name * ".tsv")
end



function all_dat_tsv()
  dats = [
  "golem_endings_tag.dat",
  "golem_sems_stem.dat",
  "golem_stems_sem.dat",
  "golem_tags_ending.dat",
  "phrases_sem.dat",
  "source.dat",
  "sem_index.dat"]
    for file in dats
        dict = load_dictionary(file)
        name = splitext(file)[1]
        save_dict1_u(dict, name * ".tsv")
    end
  println("saved ",length(dats)," *.tsv")
end



function view_sem(semnum; phrases = 5)
  #   datafile
  #sems = load_dictionary(datafile)
  #datafile = "phrases_sem.dat"; semnum=11163
  #view_sem(semnum, datafile;phrases = 6)

  sems = phrases_sem #loaded in web_utils.jl
  if haskey(sems,semnum)
  text = sems[semnum]; text = text[1:min(phrases,length(text))]; ngrams = ""
  for i in 1:length(text)
    ngrams = ngrams * text[i] * " |"
  end
  text = ngrams
  else text = "no sem=" * string(semnum)
  end
  #println(text)
  return text
end



function append_dict_u!(dict,dict_)
  # Add elements of dict_ to dict
  kk = keys(dict_)
  for k in kk
    dict[k] = get(dict,k,0) + dict_[k]
  end
end



function text_to_xml(path)
  # text files in dir=path -> one xml file in wiki-format
  #path = "C:\\articles\\fas"
  #text_to_xml(path)

  file_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru">"""
  page_prefix = "<page>\n"
  page_suffix = "</page>\n"
  file_suffix = "</mediawiki>"

  for (root, dirs, files) in walkdir(path)
                #println("Files in $root")
    fid_out = open(joinpath(root,"dump.xml"),"a+")
    xml_string = file_prefix
    write(fid_out, xml_string)
    for file in files
          #println(joinpath(root, file))
          fid_in  = open(joinpath(root,file))
          dirty = readstring(fid_in)
          close(fid_in)

          #println("replacing xml-markers ...")
          replace1 = replace(dirty,"&","&amp;")
          replace2 = replace(replace1,"<","&lt;")
          text = replace(replace2,">","&gt;")
          title      = file   #println("title=",title)
          timestamp  = "2018-12-18T12:34:56Z"
          xml_string = page_prefix *
                      "<title>" * title * "</title>" *
                      "<timestamp>" * timestamp * "</timestamp>" *
                      "<text>" * text * "</text>" * page_suffix
          write(fid_out, xml_string)
      end
      close(fid_out)
      println("total articles_count=",length(files))
  end
  return 0
end



function dir_list(path)
  # text files in dir=path -> Filename, First phrase, filesize, <br>
  #path = "C:\\articles\\fas"
  #l = dir_list(path)

  list = ""
  for (root, dirs, files) in walkdir(path)
                #println("Files in $root")
    for file in files
      if isfile(joinpath(root,file))
          fid_in  = open(joinpath(root,file))
          phrase = readstring(fid_in)
          println(joinpath(root,file))
          snippet = phrase[1:min(1080,endof(phrase))] #* " ... "
          list = list * "<b>" * file * " </b>&nbsp;&nbsp; " * snippet * "<br><br>"

                  #"&nbsp;&nbsp; filesize=" *
                  #string(filesize(joinpath(root,file))) * " <br><br>"
          close(fid_in)
        end
    end
  end
  return list
end



function sort_phrases()
  sorted = sort(collect(phrases_sem),by=x->length(x[2]))
  return sorted
end



function load_sems_clusters()
  sems_clusters = readlines(pwd()*"/sems-long/sems-long.txt")
  return sems_clusters
end



function search_sems(word)
 snippet = word * " "
 result_yes = " "
 result_no = "=>не найдено: "
 result = result_no
 found = ""
 for i in 1:length(sems_clusters)
     if contains(sems_clusters[i], word)
       # "Строка="*string(i) *
       found = "Номер кластера=" * sems_clusters[i] * "<br><br>"
       result = result_yes
       snippet = snippet * found
     end
  end
  snippet = result * snippet
 return snippet
end



function sems_to_files(limit; default=500)
  # limit=number files to create in one dir
  limit=500
  #if !isdefined(:phrases_sem)
  #    phrases_sem = load_dictionary_u("phrases_sem_1.dat")
  #end
  sorted = sort(collect(phrases_sem),by=x->length(x[2])) #,rev=true)
  #writecsv("asems", sorted)

  for i in 1:min(limit,length(sorted))
  snippet = string(sorted[i][2])
  snippet = replace(snippet,"String[",""); snippet = replace(snippet,"]","")
  snippet = replace(snippet,"\"","")
  snippet = replace(snippet,","," |")
  snippet = replace(snippet,"(",""); snippet = replace(snippet,")","");
  write(pwd()*"/sems-short/"*string(sorted[i][1]), snippet)
  end
  f = open(pwd()*"/sems-long/sems-long.txt","a+")
  for i in length(sorted):-1:length(sorted)-min(limit,length(sorted))
  snippet = string(sorted[i][2])
  snippet = replace(snippet,"String[",""); snippet = replace(snippet,"]","")
  snippet = replace(snippet,"\"","")
  snippet = replace(snippet,","," |")
  snippet = replace(snippet,"(",""); snippet = replace(snippet,")","");
  write(pwd()*"/sems-long/"*string(sorted[i][1]), snippet)
  write(f,string(sorted[i][1])*snippet * "\n")
  end
  close(f)
end



function lenta_to_topics(input_file)
  # tags, "text                ", title              , topic  ,   url
  # Рынки,"Евро может ослабнуть", Доллару предсказали, Финансы, https://lenta.ru/news/2017/03/27/eurousdequal/
  topic = "Спорт"#"Мир"
  fid_in  = open(input_file)
  io = IOBuffer(); #print(io, ) 
  lines = readlines(fid_in)
  close(fid_in)
  articles_count = n = 1
  topic_articles = t = 0
  #write(fid_out, "start")
    while n <  length(lines)
      #fields = split(lines[n], "\"")
      field1 = split(lines[n], ",")[1]
      
      fields4 = replace(lines[n], field1=>"")
      
      field3 = split(fields4, "\"")[2]
      field3 = chop(field3, head=1, tail=0) #tail?
      
      #println("n=", n, "topic=", split(fields[3], ",")[3] )
      println("n=", n )

      if split(fields[3], ",")[3] == "Мир" #"Спорт"
        text = fields[2]
        t += 1
        #println("text=", text)
        print(io, text * "\n")
      end
      n += 1
    end
  text = String(take!(io))
  text = "topic=" * topic * "  articles=" * string(t) * "\n" * text
  write("lenta_"*topic*".txt", text)
end



function lenta_to_topics2(input_file, topic::String)
  #topic = "Наука и техника" #"Спорт"#"Мир"
  fid_in  = open(input_file)
  io = IOBuffer(); #print(io, ) 
  lines = readlines(fid_in)
  close(fid_in)
  articles_count = n = 1
  topic_articles = t = 0
    while n <  length(lines)
    println("n=", n )
      if contains(lines[n], topic)
          t += 1; #println("n=", n )
          try
            field12 = split(lines[n], "\"")[2]
            #print(io, lines[n] * "\n")
            print(io, field12 * "\n")
          catch
            @warn "No content on line " * string(n) 
            sleep(2.0)
          end
      end
      n += 1
    end
  text = String(take!(io))
  text = "topic=" * topic * "  articles=" * string(t) * "\n" * text
  write("lenta_"*topic*".txt", text)
  println("articles=lines=", t, "  topic=", topic)
end



function lenta_csv_to_xml(input_file,output_file)
  #lenta_csv_to_xml(fid_in,fid_out)
  #lenta.ru csv format:  tags,"text",title,topic,url CRLF (or LF)
  #fid_in  = "/home/articles/lenta_data.csv"
  #fid_out = "/home/articles/lenta.xml"
  #fid_in  = "C:\\articles\\lenta\\lenta_data--.csv"
  #fid_out = "C:\\articles\\lenta\\lenta--.xml"

  #lenta_csv_to_xml(fid_in,fid_out)

  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  tic()
  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input csv ...")
  fid_in  = open(input_file)
  dirty = readstring(fid_in)
  close(fid_in)
  println("replacing xml-markers ...")

  replace1 = replace(dirty,"&","&amp;")
  replace2 = replace(replace1,"<","&lt;")
  dump = replace(replace2,">","&gt;")

  println("characters = ",length(dump))
  println("bytes = ",sizeof(dump))
  articles_count = 0
  pos = 1
  while pos <  sizeof(dump) #length(dump)
        #println("pos-start=",pos)
        #println("char-start=",ind2chr(dump,pos))
        tags_start  = pos
        tags_end    = searchindex(dump,",\"",pos)
        tags        = dump[tags_start:tags_end-1]
        #println("tags=",tags)
        text_start  = tags_end + 2
        text_end    = searchindex(dump,"\",",text_start)
                while dump[prevind(dump,text_end-1):text_end] == "\"\""
                text_end = searchindex(dump,"\",",nextind(dump,text_end+1))
                end
        text        = dump[text_start:text_end-1]
        #println("text=",text)
        title_start = text_end + 2
        title_end   = searchindex(dump,",",title_start)
        title       = dump[title_start:title_end-1]
        #println("title=",title)
        topic_start = title_end + 1
        topic_end   = searchindex(dump,",",topic_start)
        topic       = dump[topic_start:topic_end-1]
        #println("topic=",topic)
        url_start   = topic_end + 1
        url_end     = searchindex(dump,"\n",url_start)
                if url_end == 0
                  println("expected LF, but no more LF in file, finishing")
                  break
                end
        url         = dump[url_start:url_end]
        #println("url=",url)

        pos = url_end + 1
        #valid is <timestamp>1837-12-31T12:34:56Z</timestamp>
        #timestamp = Dates.format(now(), "dd-u-yyyy HH:MM:SS")
        timestamp = "2017-12-15T12:34:56Z"
        xml_string = page_prefix *
                    "<title>" * title * "</title>" *
                    "<article_id>" * string(articles_count+1) * "</article_id>" *
                    "<timestamp>" * timestamp * "</timestamp>" *
                    "<tags>" * tags * "</tags>" *
                    "<topic>" * topic * "</topic>" *
                    "<url>" * url * "</url>" *
                    "<text>" * text * "</text>" *
                    page_suffix
        write(fid_out,xml_string)
        articles_count += 1
        println("articles_count=",articles_count)
        if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
        end
  end
  println("total articles_count=",articles_count,"\n pos=",pos)
  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("written ",fid_out)
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  println(toc())
  return 0
end



function gz_csv_to_xml(input_file,output_file)
  #GosZakupki csv format:  short_name->title, name->text   CRLF (or LF)
  #fid_in  = "C:\\articles\\GZ\\1-52.csv"
  #fid_out = "C:\\articles\\GZ\\1-52.xml"
  #fid_in  = "C:\\articles\\GZ\\word-.csv"
  #fid_out = "C:\\articles\\GZ\\word-.xml"

  #gz_csv_to_xml(fid_in,fid_out)

  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  tic()
  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input csv ...")
  fid_in  = open(input_file)
  dirty = readstring(fid_in)
  close(fid_in)
  println("replacing xml-markers ...")

  replace1 = replace(dirty,"&","&amp;")
  replace2 = replace(replace1,"<","&lt;")
  dump = replace(replace2,">","&gt;")

  println("characters = ",length(dump))
  println("bytes = ",sizeof(dump))
  articles_count = 0
  date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
  pos = 1
  delim = "^^"
  while pos < sizeof(dump) #length(dump)
  #println("pos before=",pos)
  inn_start = pos
  inn_end = searchindex(dump,delim,pos)
  inn = dump[inn_start:inn_end-1]

  #println("inn=",inn)

  kpp_start = inn_end + 2
  kpp_end = searchindex(dump,delim,kpp_start)
  kpp = dump[kpp_start:kpp_end-1]

  title_start = kpp_end + 2
  title_end = searchindex(dump,delim,title_start)
  title = dump[title_start:title_end-1]

  text_start = title_end + 2
  text_end = searchindex(dump,delim,text_start)
  text = dump[text_start:text_end-1]

  #println("text=",text)
  time_start = text_end + 2
  #println("time_start=",time_start)

  time_end = searchindex(dump,delim,time_start) #time_start + 19
  #println("time_end=",time_end)

  time = dump[time_start:time_start+18]
  #println("time=",time)
        pos = time_end - 10
  #      println("pos after=",pos)
    #timestamp = Dates.format(Dates.unix2datetime(),date_format)
    #    timestamp = "2017-12-15T12:34:56Z"
    # date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
        xml_string = page_prefix *
                    "<title>" * "ИНН " *
                      inn * "   " * title * " " * string(articles_count+1) * "</title>" *
  #                  "<article_id>" * string(articles_count+1) * "</article_id>" *
                    "<timestamp>" * string(DateTime(time,date_format))*"Z"* "</timestamp>" *
                    "<text>" * text * "</text>" *
                    page_suffix
        write(fid_out,xml_string)
        articles_count += 1
        println("articles_count=",articles_count)
        if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
        end
        if pos <= 0
          println("EOF")
          break
        end
  end
  println("total articles_count=",articles_count,"\n pos=",pos)
  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("written ",fid_out)
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  println(toc())
  return 0
end



function gz_csv_content_to_xml(input_file,output_file)
  #GZakupki csv format:  guid+local_key->title, content->text
  #guid = 36 chars
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  tic()
  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input csv ...")
  fid_in  = open(input_file)
  dirty = readstring(fid_in)
  close(fid_in)
  println("replacing xml-markers ...")

  replace1 = replace(dirty,"&","&amp;")
  replace2 = replace(replace1,"<","&lt;")
  dump = replace(replace2,">","&gt;")

  println("characters = ",length(dump))
  println("bytes = ",sizeof(dump))
  articles_count = 0
  date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
  pos = 1
  delim = "^^^"
  while pos < sizeof(dump) #length(dump)
  #println("pos before=",pos)
  guid_start = pos
  guid_end = searchindex(dump,delim,pos)
  guid = strip(dump[guid_start:guid_end-1],'\n')

  #println("guid=",guid)

  localkey_start = guid_end + 3
  localkey_end = searchindex(dump,delim,localkey_start)
  localkey = dump[localkey_start:localkey_end-1]

  #title_start = kpp_end + 2
  #title_end = searchindex(dump,delim,title_start)
  #title = dump[title_start:title_end-1]

  text_start = localkey_end + 3
  text_end = searchindex(dump,delim,text_start)
      if text_end == 0
          text = strip(dump[text_start:Int(sizeof(dump))],'\n')
      else text = strip(dump[text_start:text_end-37],'\n')
      end


  #println("text=",text)
  #time_start = text_end + 2
  #println("time_start=",time_start)

  #time_end = searchindex(dump,delim,time_start) #time_start + 19
  #println("time_end=",time_end)

  #time = dump[time_start:time_start+18]
  #println("time=",time)
        pos = text_end-37
  #      println("pos after=",pos)
    #timestamp = Dates.format(Dates.unix2datetime(),date_format)
    #    timestamp = "2017-12-15T12:34:56Z"
    # date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
        xml_string = page_prefix *
                    "<title>" *
                    string(articles_count+100) * "IDX  " * localkey * " " *  "</title>" *
  #                  "<article_id>" * string(articles_count+1) * "</article_id>" *
                    "<timestamp>" * "2018-01-31T12:34:56Z"* "</timestamp>" *
                    "<text>" * text * "</text>" *
                    page_suffix
        write(fid_out,xml_string)
        articles_count += 1
        println("articles_count=",articles_count)
        if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
        end
        if pos <= 0
          println("EOF")
          break
        end
  end
  println("total articles_count=",articles_count,"\n pos=",pos)
  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("written ",fid_out)
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  println(toc())
  return 0
end



function benzin_csv_to_xml(input_file,output_file)
  #GZakupki csv format:  registartion_number+date->title, name->text
  #reg_num = 11 chars
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  tic()
  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input csv ...")
  fid_in  = open(input_file)
  dirty = readstring(fid_in)
  close(fid_in)
  println("replacing xml-markers ...")

  replace1 = replace(dirty,"&","&amp;")
  replace2 = replace(replace1,"<","&lt;")
  dump = replace(replace2,">","&gt;")

  println("characters = ",length(dump))
  println("bytes = ",sizeof(dump))
  articles_count = 0
  date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
  pos = 1
  delim = "^^^"
  while pos < sizeof(dump) #length(dump)
  #println("pos before=",pos)
  reg_start = pos
  reg_end = searchindex(dump,delim,pos)
  reg = strip(dump[reg_start:reg_end-1],'\n')

  #println("guid=",guid)
  #localkey_start = guid_end + 3
  #localkey_end = searchindex(dump,delim,localkey_start)
  #localkey = dump[localkey_start:localkey_end-1]
  #title_start = kpp_end + 2
  #title_end = searchindex(dump,delim,title_start)
  #title = dump[title_start:title_end-1]

  text_start = reg_end + 3
  text_end = searchindex(dump,delim,text_start)
      if text_end == 0
          text = strip(dump[text_start:Int(sizeof(dump))],'\n')
      else text = strip(dump[text_start:text_end-12],'\n')
      end
  #println("text=",text)
  #time_start = text_end + 2
  #println("time_start=",time_start)
  #time_end = searchindex(dump,delim,time_start) #time_start + 19
  #println("time_end=",time_end)
  #time = dump[time_start:time_start+18]
  #println("time=",time)
        pos = text_end-12
  #      println("pos after=",pos)
    #timestamp = Dates.format(Dates.unix2datetime(),date_format)
    #    timestamp = "2017-12-15T12:34:56Z"
    # date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
        xml_string = page_prefix *
                    "<title>" * #string(articles_count+100) * "reg_num  " *
                    reg * "</title>" *
  #                  "<article_id>" * string(articles_count+1) * "</article_id>" *
                    "<timestamp>" * "2019-01-31T12:34:56Z"* "</timestamp>" *
                    "<text>" * text * "</text>" *
                    page_suffix
        write(fid_out,xml_string)
        articles_count += 1
        println("articles_count=",articles_count)
        if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
        end
        if pos <= 0
          println("EOF")
          break
        end
  end
  println("total articles_count=",articles_count,"\n pos=",pos)
  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("written ",fid_out)
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  println(toc())
  return 0
end

  #=
  #println("vocab=",string(vocab))
  #println(words[1:min(length(words),10)],"...")
                  price_limit_small=1_000_000; price_limit_big=10_000_000;
                  price_found = -1; verdict = "Цена не найдена"
  #words = split(text)
    #for i = 1:length(words)
    #    if ismatch(r"\d(RUB)", words[i])
    #       price_found = parse(Int64,chop(chop(chop(words[i]))))
    #    end
    #end

  if     price_found == -1
       verdict = "Цена в тексте не найдена"
  elseif price_found > price_limit_small_high && categories[i] == "small_auto"
       verdict = "<font color=\"red\">Найденная цена " * string(price_found) * "RUB" *
       " превышает предел в " * string(price_limit_small_high) * " для категории " *
       category[i] * "</font><br>"
  elseif price_found < price_limit_small_low && categories[i] == "small_auto"
       verdict = "<font color=\"blue\">Найденная цена " * string(price_found) * "RUB" *
       " ниже нижнего предела в " * string(price_limit_small_low) * " для категории " *
       category[i] * "</font><br>"
  elseif price_found > price_limit_big_high && categories[i] == "big_auto"
       verdict = "<font color=\"red\">Найденная цена " * string(price_found) * "RUB" *
       " превышает предел в " * string(price_limit_big_high) * " для категории " *
       category[i] * "</font><br>"
  elseif price_found < price_limit_big_low && categories[i] == "big_auto"
       verdict = "<font color=\"blue\">Найденная цена " * string(price_found) * "RUB" *
       " ниже нижнего предела в " * string(price_limit_big_low) * " для категории " *
       category[i] * "</font><br>"
  else   verdict = "<font color=\"green\">Найденная цена " * string(price_found) * "RUB" *
       " находится в допустимых пределах для категории " *
       category[i] * "</font><br>"
end
=#



function lenta_csv_to_docs()
  #fid_in  = "C:\\articles\\FAS\\ENGINES\\engines.csv"
  #fid_out = "C:\\articles\\FAS\\ENGINES\\engines.xml"
  #fid_in  = "C:\\articles\\GZ\\word-.csv"
  #fid_out = "C:\\articles\\GZ\\word-.xml"

  #benzin_csv_to_xml(fid_in,fid_out)

  datafile = "C:\\articles\\lenta\\lenta3.csv"
  data = readstring(datafile)
  pos = 1; delim = "^^"; fnames=Dict(); articles_count = 0
    while pos < sizeof(data) #length(text)
    #  println("pos first=",pos)
      fname_start = pos
      fname_end = searchindex(data,delim,pos)
      fname = data[fname_start+1 : fname_end-2]
          #println("fname=",fname)
      fnames[fname] = 1 + get(fnames,fname,0)

      text_start = fname_end + 3
      text_end = searchindex(data,delim,text_start) - 10
      text = data[text_start:text_end-2]
          #println("text=",text)
          n = get(fnames,fname,0)
          write(fname*string(n),text); articles_count += 1
      pos = text_end +2
      if pos <= 0  println("EOF")
        break
      end
    end
  println(fnames)
  println("total articles_count=",articles_count,"\n pos=",pos)

  return 0
end



function benzin_new_csv_to_xml(input_file,output_file)
  #GZakupki csv format:  registartion_number->title, name->text
  #ЧОЙСКИЕ ЖКУ^^^270900.00^^^31704760840^^^Поставка ГСМ )бензин АИ 95
  #Заказчик: ЧОЙСКИЕ ЖКУ
  #Цена лота: 270900.00
  #Поставка ГСМ )бензин АИ 95
  #fid_in  = "C:\\articles\\FAS\\BENZIN\\b-new"
  #fid_out = "C:\\articles\\FAS\\BENZIN\\b-new.xml"
  #benzin_new_csv_to_xml(fid_in,fid_out)

  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  tic()
  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input csv ...")
  fid_in  = open(input_file)
  dirty = readstring(fid_in)
  close(fid_in)
  println("replacing xml-markers ...")

  replace1 = replace(dirty,"&","&amp;")
  replace2 = replace(replace1,"<","&lt;")
  dump = replace(replace2,">","&gt;")

  println("characters = ",length(dump))
  println("bytes = ",sizeof(dump))
  articles_count = 0
  date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:SSZ")
  pos = 1
  delim = "^^^";line_end="@@@"
  while pos < sizeof(dump) #length(dump)
  #println("pos before=",pos)
  cust_start = pos
  cust_end = searchindex(dump,delim,pos)
  cust = strip(dump[cust_start:cust_end-1],'\n')
  cust = " " * cust

  price_start = cust_end + 3
  price_end = searchindex(dump,delim,price_start)
  price = strip(dump[price_start:price_end-1],'\n')
  price = " " * price * " руб."

  reg_start = price_end + 3
  reg_end = searchindex(dump,delim,reg_start)
  reg = dump[reg_start:reg_end-1]
  reg = "№ " * reg

  text_start = reg_end + 3
  text_end = searchindex(dump,line_end,text_start)
      if text_end == 0
        break
        #   text = strip(dump[text_start:Int(sizeof(dump))],'\n')
      else text = dump[text_start:text_end-1]
      end
  #text = price * "@@@" * cust * "@@@" * text  #@@@  &lt; br &gt;

        pos = text_end+3
        xml_string = page_prefix *
                    "<title>" * #string(articles_count+100) * "reg_num  " *
                    reg * price * cust * "</title>" *
                    # * "@@" * price * cust
  #                  "<article_id>" * string(articles_count+1) * "</article_id>" *
                    "<timestamp>" * "2019-01-31T12:34:56Z"* "</timestamp>" *
                    "<text>" * text * "</text>" *
                    page_suffix
        write(fid_out,xml_string)
        articles_count += 1
        println("articles_count=",articles_count)
        if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
        end
        if pos <= 0
          println("EOF")
          break
        end
  end
  println("total articles_count=",articles_count,"\n pos=",pos)
  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("written ",fid_out)
  println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
  println(toc())
  return 0
end



function remove_fragment(input_file,output_file)
  ##=/wiki/№_31300391675__руб._ГУП_ТПО_ЖКХ_УР,0,2019-01-17
  fid_out = open(output_file,"a+")
  fid_in  = open(input_file)

  for line in eachline(fid_in)
  newline = replace(line,"/wiki/№_","")
  remove_start = searchindex(newline,"=") + 12
  #println("remove_start=",remove_start)
  remove_end = searchindex(newline,",",remove_start)
  #println("remove_end=",remove_end)
  remove = newline[remove_start:remove_end-1]
  #println("remove=",remove)
  newline2 = replace(newline,remove,"")
  #println("newline2=",newline2)
  write(fid_out,newline2)
  end
  close(fid_out)
end



function multiple_words(input_file,output_file)
  fid_in  = open(input_file)
  words = readlines(fid_in)
  close(fid_in)
  multi = []
  #println("adding uppercases ...")
        for i in 1:length(words)
          for j in 1:100
            words_clean = chomp(words[i])
            push!(multi,words_clean*" ")
          end
        end
  fid_out = open(output_file,"w")
  write(fid_out,multi)
  close(fid_out)
  return 0
end



function build_form() #(input_form,resultset)
  #result = readcsv("profile.csv";header=true)[1]
  result = DelimitedFiles.readdlm("profile.csv";header=true)[1]
  rows = size(result,1)
  names=[]; values=[]; words=[]; records=[]; all_records="" #columns=4
  lineId=[]; weights=[]; sems=[]
  record_start = """ <p><input type="checkbox" """

  for i in 1:rows #rows selected
    push!(lineId,result[i,1])
    push!(words,result[i,2])
    push!(weights,result[i,3])
    push!(sems,result[i,4])
    #println("i=",i)
    next_record = record_start *
                  "name=" * string(lineId[i]) * "  " *
                  "value=" * "\"" * string(result[i,2]) * "\"" * ">" *
                  string(words[i]) * "  " * string(weights[i]) * """</p>"""
    #println(next_record)
    push!(records,next_record)
    all_records = all_records*next_record
  end

  part_start = String(read("part_start.html"))
  part_end = String(read("part_end.html"))

  output_form = part_start * all_records * part_end
  #println(output_form)
  return output_form,rows,lineId
end



function json_to_xml(input_file,output_file)
  # Sputnik json format:  tags,"text",title,topic,url CRLF (or LF)
  # wiki-parser requires "dd-u-yyyy HH:MM:SSZ"  Z!!!
  # string(Dates.unix2datetime(1_234_567_890)) => "2009-02-13T23:31:30"
  #tic()
  #fid_in  = "/home/articles/lenta_data.csv"
  #fid_out = "/home/articles/lenta.xml"
  #fid_in  = "C:\\articles\\sputnik\\json100k"
  #fid_out = "C:\\articles\\sputnik\\json100k.xml"
  #json_to_xml(fid_in,fid_out)

  dump_prefix =
  """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/
                                    http://www.mediawiki.org/xml/export-0.10.xsd"
                version="0.10" xml:lang="ru"> """
  dump_suffix = "</mediawiki>"
  page_prefix = "<page>"
  page_suffix = "</page>\n"

  fid_out = open(output_file,"a+")
  write(fid_out,dump_prefix)

  println("reading input json ...")
  fid_in  = open(input_file)
  json_arr = readlines(fid_in)
  close(fid_in)

  articles_count = 0
  date_format = Dates.DateFormat("yyyy-mm-ddTHH:MM:SSZ")

  for i in 1:length(json_arr)
    dirty = json_arr[i]
    replace1 = replace(dirty,"&","&amp;")
    replace2 = replace(replace1,"<","&lt;")
    json_string = replace(replace2,">","&gt;")
    json_dict = JSON.parse(json_string)
    timestamp = Dates.format(Dates.unix2datetime(get(json_dict,"published_at","no such json field")),date_format)
    title = get(json_dict,"title","no such json field")
    url = get(json_dict,"url","no such json field")
    text = get(json_dict,"text","NO text")
    full_text = get(json_dict,"full_text","NO full_text")
            if full_text == "NO full_text"
            text_to_xml = text
            else
            text_to_xml = full_text
            end
    xml_string = page_prefix * "<title>" * title * "</title>" *
                "<timestamp>" * timestamp * "</timestamp>" *
                "<text>" * text_to_xml * "</text>" * page_suffix
    write(fid_out,xml_string)

    articles_count += 1
    if articles_count%1000 == 0
            println("articles_count=",articles_count)
            println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
    end
  end

  write(fid_out,dump_suffix)
  println("saving ...")
  close(fid_out)
  println("total articles_count=",articles_count)
  return 0
end



function clean_json(input_file,output_file)
    # Input: Sputnik json format:  tags,"text",title,topic,url CRLF (or LF)
    # Output: title + \n + full_text + \n
    fid_out = open(output_file,"a+")
    println("reading input json ...")
    fid_in  = open(input_file)
    json_arr = readlines(fid_in)
    close(fid_in)
    articles_count = 0

    for i in 1:length(json_arr)
      json_string = json_arr[i]
      json_dict = JSON.parse(json_string)
      title = get(json_dict,"title","no such json field")
      text = get(json_dict,"text","NO text")
      full_text = get(json_dict,"full_text","NO full_text")
              if full_text == "NO full_text"
              text_to_xml = text
              else
              text_to_xml = full_text
              end
      #clean_string = title * " \n" * text_to_xml * " \n"
      clean_string = title * " \n" #* text_to_xml * " \n"
      write(fid_out,clean_string)
      articles_count += 1
              if articles_count%1000 == 0
                println("articles_count=",articles_count)
                println(Dates.format(now(), "dd-u-yyyy HH:MM:SS"))
              end
    end
    println("saving ...")
    close(fid_out)
    println("total articles_count=",articles_count)
    return 0
end



function create_new_folder()
    # create new folder NC inside current folder with origin files
    #current_path = pwd()
    #new_folder   = joinpath(current_path,"NC")
    #arr_dirlist  = readdir()
    #exists = false
    #for f in 1:length(arr_dirlist)
    #    if arr_dirlist[f] == "NC" exists = true  end
    #end
    #if !exists
    #  mkdir(new_folder)
    #end
    #for f in 1:length(arr_dirlist)
    #cp(arr_dirlist[f],new_folder * """\\"""*arr_dirlist[f];remove_destination=true)
    #end
    #cd(new_folder)
end



function remove_comments(input_file, output_file)
  fid_out = open(output_file,"a+")
  #write(fid_out,"New version")
  fid_in  = open(input_file)
  full = readlines(fid_in)
  comments = 0
      for i in 1:length(full)
          lstripped = lstrip(full[i])
          if !contains(lstripped,"!!!ravno dies!!!")
                if  startswith(lstripped,"#")
                    #println("full[",i,"]=",full[i])
                    nocomments = ""
                        if startswith(lstripped,"!!!dies ravno!!!")
                            write(fid_out,lstripped)
                        else
                            write(fid_out,nocomments)
                            comments+=1
                        end
                else
                    removed_on_this_line_comm = split(lstripped,'#')
                    if length(removed_on_this_line_comm) > 1 #comments are in [2]
                      removed_on_this_line_comm = (split(lstripped,'#')[1]) * "\n"
                    else #no detected comments on this line
                      removed_on_this_line_comm = lstripped
                    end
                    write(fid_out,removed_on_this_line_comm)
                end
          else
              write(fid_out,lstripped)
          end
      end
    close(fid_out)
    close(fid_in)
    println("commented lines removed = ",comments)
  # return comments
end



function remove_block_comments(input_file,output_file)
  # nested multiline comments  NOT processed!
  #  #= =# on separate lines
  fid_out = open(output_file,"a+")
  #write(fid_out,"New version")
  fid_in  = open(input_file)
  full = readlines(fid_in)
  comments = 0
  comment = false

    for i in 1:length(full)

      if contains(full[i], "#=")
          comment = true
          comments+=1
      end

      if !comment
        write(fid_out,full[i])
      end

      if contains(full[i], "=#")
          comment = false
      end

    end
  close(fid_out)
  close(fid_in)
  println("block commented lines removed = ",comments)
  return comments
end



function noos(fid_in)
  d=JSON.parse(readline(fid_in))
  println("length=",length(d))
  text="";articles_count=0; fid_out = open("text","a+")
  for i in 1:length(d)
  text = d[i]["text"]
                    articles_count += 1;
                    if articles_count%1000 == 0
                    println(articles_count)
                    end
    write(fid_out,text)
  end
  close(fid_out)
end



function parse_news(fid)
  #phrases_sem = load_dictionary("phrases_sem.dat")
  a = readlines(fid) #news5")
  a_with_sems = ["a_with_sems"]; empty!(a_with_sems);
  dict_vocab = Dict(); vocab=[];
  mems_count = 0
  fname="HZ";fnames=[""];empty!(fnames)
  println("length=",length(a))
  #d=Dict("guid"=>"text"); empty!(d)
  max_sem_num = maximum(collect(keys(phrases_sem)))
  D=zeros(Int16,max_sem_num,length(a))
  println("max_sem_num=",max_sem_num)
  for i in 1:length(a)
    if length(a[i])>=65 #&& !endswith(a[i],"\r\n\r\n")
    print("i=",i);  println(" ",a[i][1:32])
    #d[string(a[i][1:32])] = chomp(string(a[i][64:end]))
  #  text = chomp(string(a[i][max(64,nextind(a[i],64)):end]))
    text = chomp(string(a[i][nextind(a[i],10):end]))
    fname = string(a[i][1:10]);push!(fnames,fname)
    sems = text_to_sems(text); mems_count = mems_count+length(sems)
    empty!(dict_vocab);  sems_text=""
      if length(sems) != 0
          for n in 1:length(sems)
            dict_vocab[sems[n]] = 1 + get(dict_vocab,sems[n],0)
          end
          vocab = sort(collect(dict_vocab), by=x->x[2],rev=true) #by=x->x[2],

          for m in 1:min(length(vocab),50)
            sems_text = sems_text * #"<br>" *
                        string(vocab[m][1]) * "=" * string(vocab[m][2]) *
                        "->" * view_sem(vocab[m][1]; phrases = 5) * "\n"
          end
      end
      push!(a_with_sems, fname *
                          " " * text * "\n" *
                          "mems=" * string(length(sems)) * "\n"  *
                          sems_text)
        #println("a_with_sems[i]=",a_with_sems[i]) #string(vocab)) #vocab
    #println(sems)
        for j in 1:length(sems)
        #println(sems[j])
        D[sems[j],i] = D[sems[j],i] + 1
        #println(D[sems[j],1])
        end
    end
  end
      println("sumD=",sum(D)); println("mems_count=",mems_count)
      #println("saving csv ...");writecsv("D",D)
  return D, a, a_with_sems, fnames
end



function select_nchar_words(fid_in)
  lines = readlines(fid_in)
  for wordlen in 3:25  
  fid_out = open("a_ru_1M_" * string(wordlen) * ".utf8","a+")

    for i in 1:length(lines)
      words = split(lines[i])
      new_line = ""
        for w in words
          if length(w) == wordlen
            new_line = new_line * w * " "
          end 
        end
        if length(new_line) > 2 
          write(fid_out, new_line * "\n")
        end
    end

  close(fid_out)
  end
end



function create_adam_texts(wcount)
  #lines = readlines(fid_in)
  idx1 = 0; idx2 = 0
  wcount = wcount#100
  #words = ["айпад","айпод"] #,"акциз","акция","афера","афиша"]
  words = ["абвгд","абвгж"] #,"акциз","акция","афера","афиша"]
  #words = ["абв1","абв2"] #,"акциз","акция","афера","афиша"]
  #words = ["аэроплан","аэропорт"]
  #words = ["аб","абв","абвг","абвгд","абвгде","абвгдеж","абвгдежз"]
  outfname = string(wcount) * "_words_" * 
             string(length(words[1])) * "_chars" * ".txt"
  rm(outfname, force=true)
  fid_out = open(outfname,"a+")
  for i in 1:wcount
      idx = rand(1:length(words))   
      if idx == 1 idx1 += 1   else idx2 += 1 end
      next_word = words[idx]
      write(fid_out, next_word * " ")
  end
  close(fid_out)
  return idx1, idx2/(length(words)-1)
end



function prepare_syns_txt()
  fid_in = open("c:/articles/RU_synonyms/syn.abramov.txt")  
  rm("syn1.txt", force=true)
  fid_out = open("syn1.txt","a+")
  lines = readlines(fid_in)
  close(fid_in) 
  for i in 1:length(lines)
    new_line = ""; new_line2 = ""; new_line3 = "";
    if i < length(lines)-1
      
      if startswith(lines[i+1], "      ")
        new_line2 = chop(lines[i+1], head=5, tail=0)
      end  
      if startswith(lines[i+2], "      ")
        new_line3 = chop(lines[i+2], head=5, tail=0)
      end  
      new_line = lines[i] * new_line2 * new_line3
      println("i=",i)
      if startswith(lines[i], "      ") ||
        split(new_line)[2] == "см."   ||
        #split(new_line)[4] == "см."   ||
        split(new_line)[2] == "||"   ||
        #split(new_line)[3] == "||"   ||
        startswith(split(new_line)[2],"[")  ||
        startswith(split(new_line)[2],"(")  ||
        !endswith( (split(new_line," ")[1]),",")
      else  write(fid_out, new_line * "\n")
      end  
    end
      if  i>30000 break end
  end
  close(fid_out)
end



function create_syn_dict()
  lines = readlines("syn1.txt") # fid_in)
  rm("syn2.txt", force=true)
  fid_out = open("syn2.txt","a+")
  words = [];
    for i in 1:length(lines)
      words = split(lines[i], ",");      println("line num=",i)
      new_line = ""
      if length(words) > 1;   #  println(words[2])
        if words[2] != "см." && words[2] != "[" && words[2] != "("
          for w in 1:length(words)
            new_line = new_line * words[w] #* ","
          end
        end 
      end
      new_line = split(new_line, ";")[1]  
      new_line = split(new_line, ".")[1]  
      new_line = split(new_line, "|")[1]  
      new_line = split(new_line, "[")[1]  
      new_line = split(new_line, "(")[1]  
      new_line = split(new_line, "см")[1]  
      #new_line = split(new_line, "")  
      if length(new_line) > 1 && length(split(new_line," ")) > 2
      write(fid_out, new_line * "\n")
      end
      if  i>30000 break end
     end
    close(fid_out)
end



  function clean_5gram(fid_in)

    alphabet = [Char.(Int('А'):Int('Е')); ['Ё'];
    Char.(Int('Ж'):Int('е')); ['ё']; Char.(Int('ж'):Int('я'))]
    #lines = readlines("5gram-10.csv") # fid_in)
    #fid_in = "5gram.csv"
    lines = readlines(fid_in)
    #rm("5gram_clean.txt", force=true)

    dict =Dict("abc" => 5 ); empty!(dict)
      for i in 1:length(lines)
        words = split(lines[i], " ");      #println("line num=",i)
        if length(words)>=5
          if startswith(words[1], alphabet) &&
            startswith(words[2], alphabet) &&
            startswith(words[3], alphabet) &&
            startswith(words[4], alphabet) &&
            startswith(words[5], alphabet) &&
            length(words[1])>2 &&
            length(words[2])>2 &&
            length(words[3])>2 &&
            length(words[4])>2 &&
            length(words[5])>2
            word5 = split(words[5],"\t")[1]
            if length(word5)>2
                new_line = ""
                for k in 1:4 #length(words)
                  new_line = new_line * words[k] * " "
                end
                new_line = new_line * word5
                #println("new_line=", new_line)
                count = get(dict, new_line, 0) + 1
                push!(dict, (new_line => count ) )
            end  
          end  
        end    
        #if  i>500000 break end
      end
      println("lines = ", length(lines))
      #sorted = sort(collect(dict),by=x->length(x[1]))
      #save_dict1_u(dict, fid_in * ".clean")
      return dict
    end
    


function batch_5gram()
  dict_big =Dict("abc" => 5); empty!(dict_big) 
      for f in readdir(pwd()*"\\5src")
        println(f)
        d=clean_5gram(pwd()*"\\5src\\"*f)
        println("length d=", length(d))
        dict_big = merge(dict_big, d)
        println("length big=", length(dict_big))
      end
      println("files = ", length(readdir(pwd()*"\\5src")))
      println("avg ", length(dict_big)/length(readdir(pwd()*"\\5src")))
  save_dict1_u(dict_big, "dict_big.clean")
end



function gen_rand_5gram()
    dict_big = load_dict1_u("dict_big.clean"; num_type=Int); println("length big=", length(dict_big))
    text = ""
    nf = sort( collect(dict_big), by=x->x[2] ) # nf=ngram_frequency [("abc", 23)]
    nf_counts = sort( collect(values(dict_big)) ) ; println("sum_nf_counts=", sum(nf_counts))
    sm = sum(nf[i][2] for i in 1:length(nf) ); println("sum=",sm)
    probs = [nf[i][2]/sm  for i in 1:length(nf) ]; 
    cum_probs = cumsum(probs); println("cum_probs=", cum_probs[1:3],"\n", cum_probs[end-3:end])
    picked_count = 0
    io = IOBuffer()
    #for n in 1:sm
    while sum(nf_counts) > 0 
      nf_num = pickone(cum_probs); 
        if nf[nf_num][2] == 0
          popat!(nf, nf_num); popat!(cum_probs, nf_num)
        else  
          picked_count +=1
          #text = text * nf[nf_num][1] * "\n"
          print(io, nf[nf_num][1] * " \n")
          nf[nf_num] = nf[nf_num][1] => nf[nf_num][2] - 1
          nf_counts[nf_num] = nf_counts[nf_num] - 1
        end  
        if picked_count%1000 == 0
          println(picked_count," picked_count (5gram processed) from ", sm)
          println("length(nf)=", length(nf) )
        end    
    end
    z = findall(!iszero, nf_counts ); println("total ngrams=", sm); println("picked=", picked_count)
    text = String(take!(io))
    write("text.txt", text )
    return z, nf, nf_counts, cum_probs
end



function pickone(cum_probs)
    i = 1; r = rand()
    while r >= cum_probs[i] && i < length(cum_probs) 
        i+=1
    end
    return i
end 



function printargs(argms...)
    println(typeof(argms))
    for (i, argm) in enumerate(argms)
    println("Arg #$i = $argm")
    end
end




function threeargs(a, b, c)
    println("a = $a::$(typeof(a))")
    println("b = $b::$(typeof(b))")
    println("c = $c::$(typeof(c))")
end



#= using UnicodePlots
function move_up(s::AbstractString)
  move_up_n_lines(n) = "\u1b[$(n)F"
  # actually string_height - 1, but we're assuming cursor is on the last line
  string_height = length(collect(eachmatch(r"\n", s)))
  print(move_up_n_lines(string_height))
  nothing
end



function animate(plots; frame_delay = 0)
    print("\u001B[?25l") # hide cursor
    for frame in frames[1:end-1]
        print(frame)
        sleep(frame_delay)
        move_up(string(frame))
    end
    print(frames[end])
    print("\u001B[?25h") # visible cursor
    nothing
end
  #125881-element Array{Tuple{Int64,Array{Float32,1},Float32},1}:
  #data_src=[ (1, [0.9, 0.25, 0.071428575, 0.355, 0.425, 0.375, 0.05, 0.35], -0.53751004),
  # (1, [0.5, 0.75, 0.5, 0.705, 0.725, 0.635, 0.65, 0.25], 0.83968353),
  # (-1, [0.9, 0.25, 0.071428575, 0.585, 0.545, 0.565, 0.15, 0.15], -0.29929376) ,
  # (-1, [0.1, 0.25, 0.64285713, 0.145, 0.245, 0.175, 0.15, 0.85], 1.801242)]

  ##data_plot = [data_src[y][3] for y in range(1, length(data_src),step = 1)]



function makeplot(curr_frame_num, data)
  lineplot(
      [x for x in range(1, length(data), step=1)],  #x-axis 
      #[data[y][3] for y in range(1, x,step = 1)], 

      #data[1 : max_x],                       #y-axis
      vcat(data[1:curr_frame_num], [i=0 for i in range(curr_frame_num+1, length(data), step = 1)] ),
      
      title = "Forecasting in Financial Markets using the ADAM Architecture",
      name = curve_name,
      xlabel = x_axis,
      ylabel = y_axis,
      color = :green,
      border = :dotted,
      canvas = DotCanvas,
      #width = max_x,
      xlim = [0, length(data)]
  )
end

  #frames = [makeplot(i, data_plot) for i in range(1, length(data_plot), step = 20)]
  #animate(frames; frame_delay = 2/length(data_plot))

  #data_res=[0,3,-3,12,13,8,11,9, 15.188426, 19.6]
  data_plot=[rand()*1.05i for i in range(1, 100, step=1)]

  curve_name = ""; x_axis = "кол-во эпох обучения";  y_axis = "% доходности";
  frames = [makeplot(i, data_plot) for i in range(1, length(data_plot), step = 20)]
  animate(frames; frame_delay = 0.5/length(data_plot))
  =#


#eop("books3_40G.txt")
#remove_syms("books3_40G.txt","ascii_72.tsv")
#eop("books3_40G_nosym.txt")^M
#sentences2lines("books3_40G_nosym_eop.txt")
#remove_punct("books3_40G_nosym_eop_lines.txt")
#remove_words("books3_40G_sentences.txt","vocab_6714250_2.tsv")
#split2sentences("books3_40G.txt")
#text_words_stat("books3_47G_en.txt")
#sentences_hash("books3_47G_en_sentences.txt")
#count_word_ngrams("books3_16G_en_cleaned_simplified.txt")

#remove_words("librusec1-16_sentences.txt","vocab_640000.tsv")
#split2sentences("librusec1-16.csv")
#text_words_stat("librusec1-16.csv")