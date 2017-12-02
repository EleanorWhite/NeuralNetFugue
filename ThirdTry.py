    
from processMidiCsv import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


TMP_MAX_LEN= 16*4
LONGEST = 16*16

CCREST = [1]*8 + [0]*12
CC_SIZE = 20



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


def goodRepTwoHot(vec):
    ''' This takes a predicted note vector of the TwoHotHorizontal form
    and converts it to correct TwoHotHorizontal form'''

    octaveVec = vec[:8]
    newOctave = [0]*8

    pitchVec = vec[8:]
    newPitch = [0]*12


    # check if the note is a rest
    # (technically, rests should always be of the form [1]*8+[0]*12, 
    # but we're only checking the [0]*12 part)
    rest = True
    for i in pitchVec:
        if i>.1:
            rest = False
    if rest:
        return [1]*8 + [0]*12

    # blindly assume that the pitch and octave are a one-hot vectors, and force 
    # them into that form by making the largest value 1 and everything else 0
    else: #not a rest
        m = max(pitchVec)
        p = pitchVec.index(m)
        newPitch[p] = 1

        m = max(octaveVec)
        o = octaveVec.index(m)
        newOctave[o] = 1 

    return newOctave + newPitch


def firstBelowInd(arr, filter):
    for i in range(len(arr)):
        if arr[i] < filter:
            return i
    return len(arr)

def goodRepCC(vec):
    ''' This takes a predicted note vector of the TwoHotHorizontal form
    and converts it to correct TwoHotHorizontal form'''

    zFilt = .3 # anything below this is a 0 and above is a 1

    octaveVec = vec[:8]
    newOctave = [0]*8

    pitchVec = vec[8:]
    newPitch = [0]*12


    # check if the note is a rest
    # (technically, rests should always be of the form [1]*8+[0]*12, 
    # but we're only checking the [0]*12 part)
    rest = True
    for i in pitchVec:
        if i> zFilt:
            rest = False
    if rest:
        return [1]*8 + [0]*12

    # blindly assume that the pitch and octave are a one-hot vectors, and force 
    # them into that form by making the largest value 1 and everything else 0
    else: #not a rest
        p = firstBelowInd(pitchVec, zFilt)
        newPitch = [1]*p + [0]*(12-p)

        o = firstBelowInd(octaveVec, zFilt)
        newOctave = [1]*o + [0]*(8-o)

    return newOctave + newPitch




def unwrap(arr):
    ''' takes an array of arrays and flattens it'''

    flatArr = []
    for a in arr:
        if type(a) == list:
            flatArr += a
        else:
            flatArr.append(a)

    return flatArr



def modelTwoHot(piece, numLines, N_values, N_epochs):

    #numLines = 4
    #N_values = 13
    max_len = TMP_MAX_LEN
    #N_epochs = 1000

    # cut the corpus into semi-redundant sequences of max_len values
    step = 3
    sentences = []
    next_values = []


    for i in range(0, len(piece) - max_len - 1, step):
        print "piece i", piece[i]

        # unwrap all note vectors
        past = []
        for note in piece[i:i+max_len]:
            past.append(unwrap(note))
        sentences.append(past)

        print "flat", unwrap(piece[i + max_len + 1])
        next_values.append(unwrap(piece[i + max_len + 1]))
    print('nb sequences:', len(sentences))

    X = np.zeros((len(sentences), max_len, numLines*N_values), dtype=np.bool)
    y = np.zeros((len(sentences), numLines*N_values), dtype=np.bool)

    # build a 2 stacked LSTM
    model = Sequential()
    #model.add(LSTM(20, return_sequences=False, input_shape=(max_len, numLines*N_values)))
    #model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=True, input_shape=(max_len, numLines*N_values)))
    model.add(Dropout(0.1))
    model.add(LSTM(30, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(numLines*N_values))
    model.add(Activation('hard_sigmoid')) # used to be softmax. consider

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(sentences, next_values, batch_size=1, epochs=N_epochs)

    return model


def modelCC(pieces, numPieces, numLines, N_values, N_epochs):

    #numLines = 4
    #N_values = 13
    max_len = TMP_MAX_LEN
    #N_epochs = 1000

    # cut the corpus into semi-redundant sequences of max_len values
    step = 16
    sentences = []
    next_values = []

    for piece in pieces:
        for i in range(0, len(piece) - max_len - 1, step):
            #print "piece i", piece[i]

            # unwrap all note vectors
            past = []
            for note in piece[i:i+max_len]:
                past.append(unwrap(note))
            sentences.append(past)

            #print "flat", unwrap(piece[i + max_len + 1])
            next_values.append(unwrap(piece[i + max_len + 1]))
        #print('nb sequences:', len(sentences))

    X = np.zeros((len(sentences), max_len, numLines*N_values), dtype=np.bool)
    y = np.zeros((len(sentences), numLines*N_values), dtype=np.bool)


    print "\n\n\nx", X
    print "\n\n\ny", y

    # make LSTM
    model = Sequential()
    model.add(LSTM(300, return_sequences=False, input_shape=(max_len, numLines*N_values)))
    model.add(Dropout(0.2))
    #model.add(LSTM(200, return_sequences=True, input_shape=(max_len, numLines*N_values)))
    #model.add(Dropout(0.1))
    #model.add(LSTM(200, return_sequences=False))
    #model.add(Dropout(0.1))
    model.add(Dense(numLines*N_values))
    model.add(Activation('hard_sigmoid')) # used to be softmax. consider

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(sentences, next_values, batch_size=1, epochs=N_epochs)

    return model


def modelCCPadding(pieces, numPieces, numLines, N_values, N_epochs):

    #numLines = 4
    #N_values = 13
    max_len = TMP_MAX_LEN
    #N_epochs = 1000

    # cut the corpus into semi-redundant sequences of max_len values
    step = 16
    sentences = []
    #x = np.array([])
    next_values = []

    samples = 0 # this will get calculated as we make things

    for piece in pieces:
        print "len pice", len(piece) - 1
        for i in range(0, min(len(piece) - 1, LONGEST), step):
            print "i", i
            #print "piece i", piece[i]

            # unwrap all note vectors
            past = []
            for note in piece[:i]:
                past += unwrap(note)

            # pad the beginning of past until it is of length LONGEST
            restTime = []
            for line in range(numLines):
                restTime.append(CCREST)
            past = unwrap(restTime)*(LONGEST-i) + past
            #print "past", past
            print "len past", len(past), LONGEST*CC_SIZE*numLines

            #print "past", past
            #print "past", unwrap(past)

            #sentences.append(past)
            sentences += past



            #print "flat", unwrap(piece[i + max_len + 1])
            #next_values.append(unwrap(piece[i]))
            next_values += unwrap(piece[i])
            print "NV", next_values

            samples += 1
        #print('nb sequences:', len(sentences))


    #X = np.zeros((len(sentences), LONGEST, numLines*N_values), dtype=np.bool)
    #y = np.zeros((len(sentences), numLines*N_values), dtype=np.bool)

    #sentence = np.reshape(sentences, (numPieces, ))
    #print "sentences", sentences

    uSentences = unwrap(sentences)
    #print "uSentences", uSentences
    #print "end uSentences"
    print "samples", samples, "len answers", len(next_values)

    x = np.array(uSentences)
    x = np.reshape(x, (samples, LONGEST, CC_SIZE*numLines))
    print "x", x

    y = np.array(next_values)
    y = np.reshape(y, (samples, CC_SIZE*numLines))

    #print "\n\n\nx", X
    #print "\n\n\ny", y

    # make LSTM
    model = Sequential()
    #model.add(LSTM(400, return_sequences=False, input_shape=(LONGEST, numLines*N_values)))
    #model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True, input_shape=(LONGEST, numLines*N_values)))
    model.add(Dropout(0.1))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(numLines*N_values))
    model.add(Activation('hard_sigmoid')) # used to be softmax. consider

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x, y, batch_size=1, epochs=N_epochs)

    return model



def trainTwoHot(N_epochs):

    filename = 'ArtOfFugueExpo.csv'
    outfile = 'out.csv'
    p = Piece()
    p.fromCSV(filename)
    
    #
    print 
    p1 = p.getTwoHotHorizontal()

    #build_model(p1, len(p1), 8)
    thsize = 20 # num ints in twoHotHorizontal
    numLines = 4
    m = modelTwoHot(p1, numLines, thsize, N_epochs)


    first = p1[0: 0 + TMP_MAX_LEN]
    first = np.reshape(first, (1,TMP_MAX_LEN,numLines*thsize))

    currentPred = first # the current four measures we're predicting off of
    fullPred = first # the full prediction

    # predict a string of 32 notes
    # TODO: for this to make sense, you need to add the past history to each of the input
    # otherwise it will always think it is predicting the second note
    lenComp = 16*4
    for i in range(lenComp):
        pred = m.predict(currentPred)
        pred = np.reshape(pred, (1,numLines*thsize))

        # put each note in a general oneHotHorizontal arrangement
        twoHotPred = []
        #for noteVec in pred:
        # go through each note vector in the prediction 
        listPred = np.ndarray.tolist(pred)
        for line in range(0,numLines*thsize,thsize):

            twoHotPred.append(goodRepTwoHot(listPred[0][line:line+thsize]))
        #fullPred.append(oneHotPred)

        newData = np.reshape(np.asarray(twoHotPred), (1,numLines*thsize))
        #newData = np.reshape(np.asarray(twoHotPred), (1,numLines,thsize))

        currentPred = np.concatenate((currentPred[0][1:], newData), axis=0)
        currentPred = np.reshape(currentPred, (1,TMP_MAX_LEN, numLines*thsize))

        fullPred = np.concatenate((fullPred[0], newData), axis=0)
        fullPred = np.reshape(fullPred, (1,TMP_MAX_LEN+i+1, numLines*thsize))

    # how we used to reshape things:
    #pred = np.reshape(pred, (1,numLines,thsize))
    fullPred = np.reshape(fullPred, (TMP_MAX_LEN+lenComp, numLines, thsize))
    fullPredArray = np.ndarray.tolist(fullPred)
    print "fullPredArray", fullPredArray

    # transform oneHotHorizontal to piece and then to csv
    predPiece = fromTwoHotHorizontal(fullPredArray)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())



def trainCC(N_epochs):

    filename = 'ArtOfFugueExpoThreeLines.csv'
    pred = Piece()
    pred.fromCSV(filename)
    predCC = pred.getCC()

    outfile = 'out.csv'

    filename = 'CBachWTC3Expo.csv'
    p1 = Piece()
    p1.fromCSV(filename)
    p1cc = p1.getCC()

    filename = 'CBachWTC9Expo.csv'
    p2 = Piece()
    p2.fromCSV(filename)
    p2cc = p2.getCC()

    filename = 'CSeventhArtOfFugueExpo.csv'
    p3 = Piece()
    p3.fromCSV(filename)
    p3cc = p3.getCC()

    filename = 'CbachFugue14Expo.csv'
    p4 = Piece()
    p4.fromCSV(filename)
    p4cc = p4.getCC()

    

    #build_model(p1, len(p1), 8)
    thsize = 20 # num ints in twoHotHorizontal
    numLines = 3
    numPieces = 4
    m = modelCC([p1cc, p2cc, p3cc, p4cc], numPieces, numLines, thsize, N_epochs)


    first = predCC[0: 0 + TMP_MAX_LEN]
    first = np.reshape(first, (1,TMP_MAX_LEN,numLines*thsize))

    currentPred = first # the current four measures we're predicting off of
    fullPred = first # the full prediction

    # predict a string of 32 notes
    lenComp = 16*4
    for i in range(lenComp):
        pred = m.predict(currentPred)
        pred = np.reshape(pred, (1,numLines*thsize))

        # put each note in a general oneHotHorizontal arrangement
        CCPred = []
        #for noteVec in pred:
        # go through each note vector in the prediction 
        listPred = np.ndarray.tolist(pred)
        for line in range(0,numLines*thsize,thsize):

            CCPred.append(goodRepCC(listPred[0][line:line+thsize]))

        newData = np.reshape(np.asarray(CCPred), (1,numLines*thsize))

        currentPred = np.concatenate((currentPred[0][1:], newData), axis=0)
        currentPred = np.reshape(currentPred, (1,TMP_MAX_LEN, numLines*thsize))

        fullPred = np.concatenate((fullPred[0], newData), axis=0)
        fullPred = np.reshape(fullPred, (1,TMP_MAX_LEN+i+1, numLines*thsize))
  

    # how we used to reshape things:
    fullPred = np.reshape(fullPred, (TMP_MAX_LEN+lenComp, numLines, thsize))
    fullPredArray = np.ndarray.tolist(fullPred)
    print "fullPredArray", fullPredArray

    # transform oneHotHorizontal to piece and then to csv
    predPiece = fromCC(fullPredArray)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())


def trainOn4(N_epochs):

    filename = 'ArtOfFugueExpoThreeLines.csv'
    pred = Piece()
    pred.fromCSV(filename)
    predCC = pred.getCC()

    outfile = 'out.csv'

    filename = 'CBachWTC3Expo.csv'
    p1 = Piece()
    p1.fromCSV(filename)
    p1cc = p1.getCC()

    filename = 'CBachWTC9Expo.csv'
    p2 = Piece()
    p2.fromCSV(filename)
    p2cc = p2.getCC()

    filename = 'CSeventhArtOfFugueExpo.csv'
    p3 = Piece()
    p3.fromCSV(filename)
    p3cc = p3.getCC()

    filename = 'CbachFugue14Expo.csv'
    p4 = Piece()
    p4.fromCSV(filename)
    p4cc = p4.getCC()

    #build_model(p1, len(p1), 8)
    thsize = 20 # num ints in twoHotHorizontal
    numLines = 3
    numPieces = 4
    m = modelCCPadding([p1cc, p2cc, p3cc, p4cc], numPieces, numLines, thsize, N_epochs)

    v = '9'

    predictStuffPadding('outAOF' + v + '.csv', predCC, numLines, thsize, m)
    predictStuffPadding('outWTC3' + v + '.csv', p1cc, numLines, thsize, m)
    predictStuffPadding('outWTC9' + v + '.csv', p2cc, numLines, thsize, m)
    predictStuffPadding('outSAOF' + v + '.csv', p3cc, numLines, thsize, m)
    predictStuffPadding('outFF' + v + '.csv', p4cc, numLines, thsize, m)




def predictStuffPadding(outfile, predCC, numLines, thsize, m):


    first = predCC[0: 0 + TMP_MAX_LEN]
    first = np.reshape(first, (1,TMP_MAX_LEN,numLines*thsize))

    currentPred = first # the current four measures we're predicting off of

    # pad currentPred
    restTime = []
    for line in range(numLines):
        restTime.append(CCREST)
    #rtnp = np.array(unwrap(restTime*(LONGEST - len(currentPred))))
    #rtnp = np.reshape(rtnp, (1, LONGEST-len(currentPred), CC_SIZE*numLines))
    print "restTime", restTime
    #print "currentPred", currentPred
    flatCurrentPred = unwrap(np.ndarray.tolist(currentPred[0]))
    print "flat curr pred len", len(flatCurrentPred)
    flatRestTime = unwrap(unwrap(restTime)*(LONGEST-TMP_MAX_LEN))
    print "flat rest len", len(flatRestTime)
    currentPred = np.array(flatRestTime + flatCurrentPred)
    print "currentPred padded", currentPred
    currentPred = np.reshape(currentPred, (1,LONGEST, CC_SIZE*numLines))

    #currentPred = np.concatenate((rtnp, currentPred), axis = 1)

    print currentPred
    fullPred = first # the full prediction

    # predict a string of 32 notes
    lenComp = 16*4
    for i in range(lenComp):
        pred = m.predict(currentPred)
        pred = np.reshape(pred, (1,numLines*thsize))

        # put each note in a general oneHotHorizontal arrangement
        CCPred = []
        #for noteVec in pred:
        # go through each note vector in the prediction 
        listPred = np.ndarray.tolist(pred)
        for line in range(0,numLines*thsize,thsize):

            CCPred.append(goodRepCC(listPred[0][line:line+thsize]))

        newData = np.reshape(np.asarray(CCPred), (1,numLines*thsize))

        currentPred = np.concatenate((currentPred[0][1:], newData), axis=0)
        currentPred = np.reshape(currentPred, (1,LONGEST, numLines*CC_SIZE))

        fullPred = np.concatenate((fullPred[0], newData), axis=0)
        fullPred = np.reshape(fullPred, (1,TMP_MAX_LEN+i+1, numLines*thsize))
  

    # how we used to reshape things:
    fullPred = np.reshape(fullPred, (TMP_MAX_LEN+lenComp, numLines, thsize))
    fullPredArray = np.ndarray.tolist(fullPred)
    #print "fullPredArray", fullPredArray

    # transform oneHotHorizontal to piece and then to csv
    predPiece = fromCC(fullPredArray)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())



def predictStuff(outfile, predCC, numLines, thsize, m):


    first = predCC[0: 0 + TMP_MAX_LEN]
    first = np.reshape(first, (1,TMP_MAX_LEN,numLines*thsize))

    currentPred = first # the current four measures we're predicting off of
    fullPred = first # the full prediction

    # predict a string of 32 notes
    lenComp = 16*4
    for i in range(lenComp):
        pred = m.predict(currentPred)
        pred = np.reshape(pred, (1,numLines*thsize))

        # put each note in a general oneHotHorizontal arrangement
        CCPred = []
        #for noteVec in pred:
        # go through each note vector in the prediction 
        listPred = np.ndarray.tolist(pred)
        for line in range(0,numLines*thsize,thsize):

            CCPred.append(goodRepCC(listPred[0][line:line+thsize]))

        newData = np.reshape(np.asarray(CCPred), (1,numLines*thsize))

        currentPred = np.concatenate((currentPred[0][1:], newData), axis=0)
        currentPred = np.reshape(currentPred, (1,TMP_MAX_LEN, numLines*thsize))

        fullPred = np.concatenate((fullPred[0], newData), axis=0)
        fullPred = np.reshape(fullPred, (1,TMP_MAX_LEN+i+1, numLines*thsize))
  

    # how we used to reshape things:
    fullPred = np.reshape(fullPred, (TMP_MAX_LEN+lenComp, numLines, thsize))
    fullPredArray = np.ndarray.tolist(fullPred)
    print "fullPredArray", fullPredArray

    # transform oneHotHorizontal to piece and then to csv
    predPiece = fromCC(fullPredArray)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())




def trainFullFugue(N_epochs):

    filename = 'ArtOfFugueExpoThreeLines.csv'
    pred = Piece()
    pred.fromCSV(filename)
    predCC = pred.getCC()

    outfile = 'out.csv'

    filename = 'CBachWTC3Expo.csv'
    p1 = Piece()
    p1.fromCSV(filename)
    p1cc = p1.getCC()

    






# ------------------------------
# BRAZENLY STOLEN FROM DEEPJAZZ
# ------------------------------

'''def main(args):


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
    predPieceCsv.write(predPiece.csv())'''
        


def main(args):
    #a = [0.8633742928504944, 0.8520313501358032, 0.6155416965484619, 0.8277215957641602, 0.7745899558067322, 0.6654524803161621, 0.5017514824867249, 0.5754479169845581, 0.3196893334388733, 0.364773690700531, 0.2010064721107483, 0.3020211458206177, 0.21267035603523254, 0.22699597477912903, 0.23443511128425598, 0.2467953860759735, 0.32214605808258057, 0.1710146963596344, 0.18880566954612732, 0.21740081906318665]
    #print goodRepCC(a)
    trainOn4(250)





''' If run as script, execute main '''
if __name__ == '__main__':
    import sys
    main(sys.argv)

# ------------------------------
# END
# ------------------------------
