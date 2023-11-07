function mark_sentences(input_file, d_words, d_sentences)
    # mark full sentences, count known and unknown words in "_processed.txt" files
    split2sentences(input_file)
    input_file = split(input_file, ".")[1] * "_sentences.txt"
    text = read(input_file, String)
    arr = split(text, '\n')
    output_file = split(input_file, ".")[1] * "_marked.txt"
    io = IOBuffer()
    count_sentences = length(arr)
    count_matched = 0
    unknown = 0; known = 0; all_words = 0 # count words
    d_unknown = Dict{String, Int64}()
    for sentence in arr
        marked = sentence
        if sentence =="" || startswith(sentence,"»тер") || startswith(sentence,"—генер")
            marked = '\n'
        else
            words = split(sentence)
            all_words += length(words)
            for word in words
                if get(d_words, word, 0) == 0 
                    unknown += 1
                    d_unknown[word] = get(d_unknown, word, 0) + 1 
                else known += 1  
                end 
            end

            count = get(d_sentences, sentence, 0)
            marked = sentence * "[" * string(count) * "] " * '\n' 
            if count > 0
                count_matched += 1
            end    
        end    
        print(io, marked)
    end    
    marked_text = String(take!(io))
    write(output_file, marked_text)
    
    ratio_sentences = round(100 * count_matched / count_sentences, digits=2) 
    println(ratio_sentences, " % full sentences")
    rep_sent = string(ratio_sentences) * " % full sentences\n"
    
    ratio_words = round(100 * unknown / all_words, digits=2)
    println(ratio_words, " % known words")
    rep_word = string(ratio_words) * " % known words\n"
    
    io = open(output_file, "a")
    write(io, rep_sent * rep_word)
    close(io)
    # return d_unknown #count_sentences
end