function load_tsv(words, sentences)
    # d_words, d_sentences = load_tsv("words_stat.tsv", "sentences_stat.tsv")
    # d_words, d_sentences = load_tsv("TS_nq_words_stat.tsv", "TS_nq_sentences_stat.tsv")
    d_words = load_dict1(words)
    d_sentences = load_dict1(sentences)
    return d_words, d_sentences
end 