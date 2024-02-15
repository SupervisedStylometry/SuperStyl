
def count_process(args, embeddedFreqs=False):
    if embeddedFreqs:
        key = "embedded"
    else:
        key = "wordCounts"

    text, feat_list = args
    local_freqs = list([0] * len(feat_list))
    for word, value in text[key].items():
        if word in feat_list:
            local_freqs[feat_list.index(word)] = value
    return text, local_freqs
