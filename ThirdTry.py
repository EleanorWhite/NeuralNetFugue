    
from processMidiCsv import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


TMP_MAX_LEN= 1

# TODO: figure out how to parse the CSV midi files

# ------------------------------
# BRAZENLY STOLEN FROM DEEPJAZZ
# ------------------------------
def build_model(corpus, val_indices, max_len, N_epochs=128):
    # number of different values or words in corpus
    N_values = len(set(corpus))

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []
    # the model is trained to take a sequence of max_len, and return a sequence
    # that is shifted over 3 notes -- most of it is just parroting back
    for i in range(0, len(corpus) - max_len, step):
        sentences.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print('nb sequences:', len(sentences))

    # transform data into binary matrices
    X = np.zeros((len(sentences), max_len, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), N_values), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, val in enumerate(sentence):
            X[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1

    # build a 2 stacked LSTM
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, N_values)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(N_values))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(X, y, batch_size=128, nb_epoch=N_epochs)

    return model

    # ------------------------------
    # END
    # ------------------------------


def goodRep(vec):
    ''' This takes a predicted note vector of the oneHotHorizontal form
    and converts it to correct oneHotHorizontal form'''

    octave = int(round(vec[0])) # round to nearest int

    pitchVec = vec[1:]
    newPitch = [0]*12
    # check if a note is playing
    rest = True
    for i in pitchVec:
        if i > .1:
            rest = False

    # blindly assume that this is a one-hot vector, and force it into that form
    # by making the largest value 1 and everything else 0
    if not rest:
        m = max(pitchVec)
        p = pitchVec.index(m)
        newPitch[p] = 1
    return [octave] + newPitch




def unwrap(arr):
    ''' takes an array of arrays and flattens it'''

    flatArr = []
    for a in arr:
        flatArr += a

    return flatArr

def model(piece):

    numLines = 4
    N_values = 13
    max_len = TMP_MAX_LEN
    N_epochs = 1000

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []

    for i in range(0, len(piece) - 2*max_len, step):
        print "piece i", piece[i]
        sentences.append(piece[i])
        print "flat", unwrap(piece[i + 1])
        next_values.append(unwrap(piece[i + 1]))
    print('nb sequences:', len(sentences))

    X = np.zeros((len(sentences), numLines, N_values), dtype=np.bool)
    y = np.zeros((len(sentences), numLines*N_values), dtype=np.bool)

    print "\n\n\nx", sentences
    print "\n\n\ny", next_values

    #for i, sentence in enumerate(sentences):
    #    for t, val in enumerate(sentence):
    #        X[i, t, val_indices[val]] = 1
    #    y[i, val_indices[next_values[i]]] = 1


    #print "\n\n\nx", X
    #print "\n\n\ny", y

    # build a 2 stacked LSTM
    model = Sequential()
    model.add(LSTM(28, return_sequences=False, input_shape=(numLines, N_values)))
    model.add(Dropout(0.2))
    #model.add(LSTM(128, return_sequences=True, input_shape=(numLines, N_values)))
    #model.add(Dropout(0.2))
    #model.add(LSTM(128, return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(numLines*N_values))
    model.add(Activation('hard_sigmoid')) # used to be softmax. consider

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(sentences, next_values, batch_size=1, epochs=N_epochs)

    return model









# ------------------------------
# BRAZENLY STOLEN FROM DEEPJAZZ
# ------------------------------

def main(args):


    try:
        N_epochs = int(args[1])
    except:
        N_epochs = 128 # default

    filename = 'ArtOfFugueExpo.csv'
    outfile = 'out.csv'
    p = Piece()
    p.fromCSV(filename)
    
    #
    print 
    p1 = p.getOneHotHorizontal()

    #build_model(p1, len(p1), 8)
    m = model(p1)

    first = p1[0: 0 + TMP_MAX_LEN]
    first = np.reshape(first, (1,4,13))
    print "\n\n first", first
    pred = m.predict(first)
    pred = np.reshape(pred, (1,4,13))
    print "\n\n\n prediction", pred


    # add first prediction to general prediction list
    fullPred = []
    oneHotPred = []
    for noteVec in pred[0]:
        oneHotPred.append(goodRep(np.ndarray.tolist(noteVec)))
    fullPred.append(oneHotPred)

    # predict a string of 32 notes
    for i in range(16*4):
        pred = m.predict(pred)
        pred = np.reshape(pred, (1,4,13))

        # force them to conform to the right pitch rep for oneHotHorizontal
        for noteVec in pred[0]:
            print goodRep(np.ndarray.tolist(noteVec))
        print "end measure"

        # put each note in a general oneHotHorizontal arrangement
        oneHotPred = []
        for noteVec in pred[0]:
            oneHotPred.append(goodRep(np.ndarray.tolist(noteVec)))
        fullPred.append(oneHotPred)


    print "full pred", fullPred
    # transform oneHotHorizontal to piece and then to csv
    predPiece = fromOneHotHorizontal(fullPred)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())
        





''' If run as script, execute main '''
if __name__ == '__main__':
    import sys
    main(sys.argv)

# ------------------------------
# END
# ------------------------------