def filter_diff_lines(str):
    # remove everything that isn't an added or removed line.
    lines = str.split("\n")
    results = []
    for line in lines:
        length = len(line)
        if length > 0:
            if line[0] == "+" or line[0] == "-":
                if length == 1 or ( line[1] != "+" and line[1] != "-"):
                    if line[0] == "+":
                        line = " LINEADDEDTOKEN " + line[1:]
                    elif line[0] == "-":
                        line = " LINEREMOVEDTOKEN " + line[1:]
                    results.append(line)
            elif line[:10] == "diff --git":
                results.append(" NEWFILETOKEN ")
    return " ".join(results)


def tokenize(text, lower=True):
    ''' Tokenizes code. All consecutive alphanumeric characters are grouped into one token.
    But we also split on camel case
    Thereby trying to heuristically match identifiers.
    All other symbols are seen as one token.
    Whitespace is stripped.
    '''
    seq = []
    curr = ""
    for c in text:
        if c.isalnum():
            # this is a camelCase variable, so start a new token
            if c.isupper() and curr != "" and not curr[-1].isupper():
                seq.append(curr)
                curr = ""
            curr += c
        else:
            if curr != "":
                seq.append(curr)
                curr = ""
            if not c.isspace():
                seq.append(c)
    if curr != "":
        seq.append(curr)
    return [_f.lower() for _f in seq if _f]
