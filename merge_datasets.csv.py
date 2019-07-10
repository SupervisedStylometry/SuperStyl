import pandas

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', help="output file name", type=str)
    parser.add_argument('paths', nargs='+', action='store', help="Path to test file")
    args = parser.parse_args()

    out = None

    for path in args.paths:
        data = pandas.read_csv(path, index_col=0)

        if out is None:
            metadata = data.loc[:, ['author', 'lang']]
            out = data.drop(['author', 'lang'], axis=1)

        else:
            out = pandas.concat([out, data], axis=1)

    out = pandas.concat([metadata, out], axis=1).to_csv(args.o)