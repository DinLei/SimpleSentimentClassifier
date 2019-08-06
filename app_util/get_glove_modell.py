import shutil
import platform
from gensim.models import KeyedVectors


def construct_model(filename, model_name="glove_twitter_200d", save_path=None):
    # More models can be downloaded from http://nlp.stanford.edu/projects/glove/
    # glove_file="glove.840B.300d.txt"

    glove_file = filename

    file_info = glove_file.split(".")
    pla = platform.platform().split('-')[0]

    num_lines = get_file_rows(filename)
    dims = int(file_info[-2].strip('d'))

    # # Output: Gensim Model text format.
    gensim_file = model_name + '.txt'
    gensim_first_line = "{} {}".format(num_lines, dims)
    #
    # # Prepends the line.
    if pla == "linux" or pla == "linux2":
        prepend_line(glove_file, gensim_file, gensim_first_line)
    else:
        prepend_slow(glove_file, gensim_file, gensim_first_line)

    model = KeyedVectors.load_word2vec_format(gensim_file, binary=False)
    print("glove model constructed successfully!")

    if save_path:
        model.save(save_path + "/" + model_name)
        print("glove model has been saved, which name is",
              model_name, "save path is", save_path)

    return model


def get_file_rows(filename):
    with open(filename, 'r') as f:
        count = len(f.readlines())
    return count


def prepend_line(infile, outfile, line):
    """
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load_model_from_local(path):
    return KeyedVectors().load(path)


if __name__ == "__main__":
    glo_file = "./corpus/glove.twitter.27B.200d.txt"
    save_file = "./glov_model"

    construct_model(glo_file, save_path=save_file)
    # model = load_model(glo_file, save_path='/home/dinglei/桌面/corpus/')
    #
    # word_list = ["woman", u'country']
    #
    # for word in word_list:
    #     print(word,'--')
    #     for i in model.most_similar(word, topn=10):
    #         print(i[0], i[1])
    #     print("")
