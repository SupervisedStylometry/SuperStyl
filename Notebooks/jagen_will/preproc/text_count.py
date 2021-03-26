
def count_process(args):
    text, feat_list = args
    local_freqs = list([0] * len(feat_list))
    for word, value in text["wordCounts"].items():
        if word in feat_list:
            local_freqs[feat_list.index(word)] = value
    return text, local_freqs
