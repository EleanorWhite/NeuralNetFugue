    
from processMidiCsv import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN


TMP_MAX_LEN= 16*4
LONGEST = 16*16

CCREST = [1]*8 + [0]*12
CC_SIZE = 20

MAX_SECT_LEN = 16*6 # maximum length of a section is 6 measures


def firstBelowInd(arr, filter):
    for i in range(len(arr)):
        if arr[i] < filter:
            return i
    return len(arr)

def goodRepCC(vec):
    ''' This takes a predicted note vector of the CC form
    and converts it to correct CC form'''

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


def modelRecurrent(x,y, numPieces, numLines, N_epochs):
    ''' This makes a model that takes in a section of a fugue (padded out to be 
    MAX_SECT_LEN long) and creates another one'''

    max_len = TMP_MAX_LEN

    # make LSTM
    model = Sequential()
    model.add(SimpleRNN(10, return_sequences=True, input_shape=(1, CC_SIZE*numLines*MAX_SECT_LEN)))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(10, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(numLines*CC_SIZE*MAX_SECT_LEN))
    model.add(Activation('hard_sigmoid')) # used to be softmax. consider

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.fit(x, y, batch_size=1, epochs=N_epochs)

    return model


def getPieceSections(filename, firstLen, secondLen, thirdLen, numLines):
    ''' This takes in a file and creates an array in CC with the first and second
    sections, and the second and third sections
    '''

    p = Piece()
    p.fromCSV(filename)
    pcc = p.getCC()

    restTime = []
    for line in range(numLines):
        restTime.append(CCREST)

    first = pcc[0:firstLen] + [restTime]*(MAX_SECT_LEN-firstLen)
    second = pcc[firstLen: firstLen+secondLen] + [restTime]*(MAX_SECT_LEN-secondLen)
    third = pcc[firstLen+secondLen: firstLen+secondLen +thirdLen]+ [restTime]*(MAX_SECT_LEN-thirdLen)

    x = [first, second]
    y = [second,third]

    return x,y


def trainOn4(N_epochs):
    ''' This trains a model on 4 fugues and outputs an example fugue on 6 subjects
    (4 known, 2 unknown)
    '''

    numLines = 3
    numPieces = 4
    thsize = 20 # num ints in twoHotHorizontal

    #NOT BEING TRAINED ON
    filename = 'CArtOfFugueExpoThreeLines.csv'
    x1,y1 = getPieceSections(filename, 16*4, 16*4, 16*4, numLines)
    

    outfile = 'out.csv'

    filename = 'CBachWTC3Expo.csv'
    x2,y2 = getPieceSections(filename, 16*2+6, 16*2, 16*2, numLines)

    filename = 'CBachWTC9Expo.csv'
    x3,y3 = getPieceSections(filename, 6*4, 6*5, 6*4, numLines) # this one is in 3/8

    filename = 'CSeventhArtOfFugueExpo.csv'
    x4,y4 = getPieceSections(filename, 8*5, 8*5, 8*4+2, numLines) # this one is in 2/4

    filename = 'CbachFugue14Expo.csv'
    x5,y5 = getPieceSections(filename, 16*3, 16*3 + 6+14, 16*3+2, numLines)

    # NOT BEING TRAINED
    filename = 'BachWTC1ExpoThreeLines.csv'
    x6,y6 = getPieceSections(filename, 8*3, 8*3, 8*4, numLines)


    x = []
    # concatenate all the arrays
    for i in [x2,x3,x4,x5]:
        x += unwrap(unwrap(unwrap(i)))
        print "len single", len(unwrap(unwrap(unwrap(i))))


    y = []
    # concatenate all the arrays
    for i in [y2,y3,y4,y5]:
        y += unwrap(unwrap(unwrap(i)))


    x_tot = np.array(x)
    x_tot = np.reshape(x_tot, ( numPieces*(numLines-1),1,  CC_SIZE*numLines*MAX_SECT_LEN))
    y_tot = np.array(y)
    y_tot = np.reshape(y_tot, (numPieces*(numLines-1), CC_SIZE*numLines*MAX_SECT_LEN))


    
    m = modelRecurrent(x_tot,y_tot,numPieces,numLines, N_epochs)

    v = '16'


    predictStuff('outAOF' + v + '.csv', x1[0], x1[1], numLines, m)
    predictStuff('outWTC3' + v + '.csv', x2[0], x2[1], numLines, m)
    predictStuff('outWTC9' + v + '.csv', x3[0], x3[1], numLines, m)
    predictStuff('outSAOF' + v + '.csv', x4[0], x4[1], numLines, m)
    predictStuff('outFF' + v + '.csv', x5[0], x5[1], numLines, m)
    predictStuff('outWTC1' + v + '.csv', x6[0], x6[1], numLines, m)



def predictStuff(outfile, first, second, numLines, model):
    ''' This writes a fugue that is predicted based on the first and second
    sections of the fugue. It write this to outfile.
    '''


    first = np.array([unwrap(unwrap(unwrap(first)))])
    first = np.reshape(first, (1, 1,  CC_SIZE*numLines*MAX_SECT_LEN))
    second = np.array([unwrap(unwrap(unwrap(second)))])
    second = np.reshape(second, (1, 1,  CC_SIZE*numLines*MAX_SECT_LEN))

    pred = model.predict(first)

    # make it a standard shape
    pred = np.reshape(pred, (MAX_SECT_LEN,numLines, CC_SIZE))
    predList = np.asarray(pred)
    CCpred = []
    for time in predList:
        t = []
        for note in time:
            t.append(goodRepCC(note))
        CCpred.append(t)


    pred2 = model.predict(second)

    # make it a standard shape
    pred2 = np.reshape(pred2, (MAX_SECT_LEN,numLines, CC_SIZE))
    predList2 = np.asarray(pred2)
    CCpred2 = []
    for time in predList2:
        t = []
        for note in time:
            t.append(goodRepCC(note))
        CCpred2.append(t)


    subj = np.reshape(first, (MAX_SECT_LEN,numLines, CC_SIZE))

    fullPred = deletePadding(np.ndarray.tolist(subj)) + deletePadding(CCpred) + deletePadding(CCpred2)


    predPiece = fromCC(fullPred)
    predPieceCsv = open(outfile, 'w')
    predPieceCsv.write(predPiece.csv())


def deletePadding(piece):
    ''' Takes out all leading rests that is a rest in all lines '''
    for i in range(len(piece) -1, 0, -1):
        print "PIECE", piece
        for note in piece[i]:
            print (note == CCREST)
            if not(note == CCREST):
                return piece[:i+1]



def main(args):
    trainOn4(400)



if __name__ == '__main__':
    import sys
    main(sys.argv)

