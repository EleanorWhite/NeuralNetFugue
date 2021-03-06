# Eleanor White, Nov 7, 2017

import csv


HALF_NOTE = 960
SIXTEENTH_NOTE = 120

TRACK = 0
TIME = 1
TYPE = 2
PITCH = 4
VELOCITY = 5


MIDI_CONVERSION = 120# about one 16th note happens every 4 time units in midi
MIDI_SWITCH = 4 #amount of silence in midi note-switching


START_HEADERS = '0, 0, Header, 1, 4, 480\n'
START_HEADERS_LIST = ['0','0','Header', '1', '4', '480']
STANDARD_HEADERS ='1, 0, Start_track\n\
1, 0, Time_signature, 2, 1, 24, 8\n\
1, 0, Key_signature, 4, "major"\n\
1, 0, Tempo, 454545\n'
# TODO: fix to not break endtime
TRACK_END_HEADERS = '1, 30960, End_track'
TRACK_END_HEADERS_LIST = ['1', '30960', 'End_track']


END_HEADERS_LIST = ['0', '0', 'End_of_file']

def genStartHeaders():
	return '0, 0, Header, 1, 4, 480\n'

def genStandardHeaders(trackNum):
	s = trackNum + ', 0, Start_track\n' \
	+ trackNum + ', 0, Time_signature, 2, 1, 24, 8\n'\
	+ trackNum + ', 0, Key_signature, 4, "major"\n'\
	+ trackNum +  ', 0, Tempo, 454545\n'


# TODO: consider whether you really want this time suig to be 2/2
def genStandardHeadersList(trackNum):
	return [[trackNum, '0', 'Start_track'], \
	[trackNum, '0', 'Time_signature', '2', '1', '24', '8'], \
	[trackNum, '0', 'Key_signature', '4', "major"], \
	[trackNum, '0', 'Tempo', '454545']]

def genEndHeadersList(trackNum, time):
	return [trackNum, time, 'End_track']


#filename = 'bach.csv'

class Note:

	def __init__(self,arr):

		verbose("Creating note object")

		self.track = int(arr[0])
		self.time = int(arr[1])
		self.type = arr[2].strip()
		self.channel = int(arr[3])
		self.pitch = int(arr[4])
		self.velocity = int(arr[5])

	def toArr(self):
		return [self.track, self.time, self.type, self.channel, self.pitch, self.velocity]

	def toStr(self):
		return str(self.track) + ", " + str(self.time) + ", " + str(self.type) + ", " \
		+ str(self.channel) + ", " + str(self.pitch) + ", " + str(self.velocity)


def noteWithPitch(pitch, trackNum, time, on):
    ''' creates a note with specified value'''
    sel = Note(['0']*6)

    sel.pitch = pitch
    sel.track = trackNum
    sel.time = time

    # defaults
    sel.type = "Note_on_c"
    sel.channel = 0
    if (on): sel.velocity = 80
    else: sel.velocity = 0 # stop note

    return sel



class Track:

	def __init__(self, arr):

		verbose("Creating track object")

		headers = []
		notes = []
		channelEvents = []
		for row in arr:

			# all note rows have "Note_on_c" as the type
			if row[TYPE].strip() == "Note_on_c":
				newNote = Note(row)
				notes.append(newNote)
			elif row[TYPE].strip() == "Program_c" or row[TYPE].strip() == "Control_c":
				channelEvents.append(row)
			else:
				headers.append(row)
		self.headers = headers # this is an array of the header rows
		self.notes = notes # this is an array of notes
		self.channelEvents = channelEvents # this is an array on non-note channel events




	def getEndTrack(self):
		''' This returns the time at which the track ends. It assumes that 
		there is an End_track header somewhere'''
		for i in self.headers:
			if i[TYPE].strip() == "End_track":
				return int(i[TIME])
		return None

	def getLatestBeforeTime(self, time):
		''' This returns the last note that has an event before/during the current timestep.
		If there are no rests, this will return the note that is currently playing.
		'''

		# if the line hasn't started yet, return none
		if time > self.getEndTrack():
			print "ERROR: Time is after end of track"
			return None

		for note in range(len(self.notes)-1):

			# This is before the first note, so nothing is playing
			if int(self.notes[0].time) > time:
				return None

			if int(self.notes[note+1].time) > time:
				# If it's a rest, return None
				if self.notes[note].velocity == 0:
					return None
				# return the Note object
				return self.notes[note]

		return self.notes[-1]


	def getTrackNum(self):
		return self.notes[0].track


class Piece:


	def __init__(self):
		''' Creates a sketch default piece object '''

		verbose("Creating default Piece object")

		self.head = None
		self.endFile = None
		self.tracks = []


	def fromCSV(self,filename):
		''' Reads in a piece from the output of the midicsv program '''

		verbose("Reading in Piece from file " + filename)
		
		trackArrs = []
		tracks = []
		file = open(filename)
		csvFile = csv.reader(file)
		for row in csvFile:

			# only works because sorted by track
			# make sure row[1] is an int, not a string
			trackNum = int(row[0])

			# keep track of the two special headers
			if trackNum == 0:
				if row[TYPE].strip() == "Header":
					self.head = row
				if row[TYPE].strip() == "End_of_file":
					self.endFile = row
			# make arrays symbolizing every track
			else:
				if trackNum > len(trackArrs):
					trackArrs.append([])
				trackArrs[trackNum-1].append(row)


		# create Track objects for all the tracks
		# the index of the track is trackNum-1
		for track in trackArrs:
			t = Track(track)
			tracks.append(t)

		self.tracks = tracks


	def getHorizontal(self):
		''' Gives back an array of arrays representing each note at 16-note time intervals'''

		verbose("Getting Horizontal form from Piece object")

		time = 0
		notes = []

		endTime = self.tracks[0].getEndTrack()

		while time <= endTime:
			currNote = [] # an array of the pitch for each line
			for track in self.tracks:
				latestNote = track.getLatestBeforeTime(time)

				# If the note is a rest in this line, it is 0
				if latestNote == None: 
					currNote.append(0)
				else:
					currNote.append(track.getLatestBeforeTime(time).pitch)
			notes.append(currNote)			

			time += int(SIXTEENTH_NOTE)

		return notes

	def getOneHotHorizontal(self):
		''' This one-hot encoding is going two have a one-hot vector for the pitch in an octave, 
		and the first value in the vector is the octave'''

		verbose("Getting OneHotHorizontal form from Piece object")

		hor = self.getHorizontal()

		outList = []

		for time in hor:
			newTime = []
			for line in time:
				# convert pitch int into a one hot vector
				pitch = line%12
				pitchVec = [0]*12
				pitchVec[pitch] = 1

				octave = line/12

				newTime.append([octave] + pitchVec)
			outList.append(newTime)
		return outList

	def getTwoHotHorizontal(self):
		''' This one-hot encoding is going two have a one-hot vector for the pitch in an octave, 
		and the first value in the vector is the octave'''

		verbose("Getting TwoHotHorizontal form from Piece object")

		hor = self.getHorizontal()

		outList = []


		for time in hor:
			newTime = []
			for line in time:
				# if it is a rest, zero everything out
				if line == 0:
				    newTime.append([1]*8 + [0]*12)
				else:
					# convert pitch int into one hot vector
					pitch = line%12
					pitchVec = [0]*12
					pitchVec[pitch] = 1
					# convert octave int into one hot vector
					octave = line/12
					octaveVec = [0]*8
					octaveVec[octave] = 1
					newTime.append(octaveVec + pitchVec)
			outList.append(newTime)

		return outList

	def getCC(self):
		''' This is an encoding much like twoHotHorizontal, but instead of having a one-hot vector
		it is ones until the index where it is normally 1, at which point it switches to 0 '''

		verbose("Getting CC form from Piece Object")

		hor = self.getHorizontal()

		outList = []


		for time in hor:
			newTime = []
			for line in time:
				#print "line", line
				# if it is a rest, zero everything out
				if line == 0:
				    newTime.append([1]*8 + [0]*12)
				else:
					# convert pitch int into one hot vector
					pitch = line%12 + 1
					pitchVec = [1]*pitch + [0]*(12-pitch)
					#print "pitchVec", pitchVec
					# convert octave int into one hot vector
					octave = line/12 + 1
					octaveVec = [1]*octave + [0]*(8-octave)
					#print "newLine", octaveVec + pitchVec
					newTime.append(octaveVec + pitchVec)

			outList.append(newTime)

		return outList


	def arrToCsv(self, arr):
		s = ""
		for i in arr:
			s += str(i)
			s += ", "
		# get rid of the last comma and space
		return s[:-2]


	def csv(self):
		''' This takes in a Piece object and returns the csv representation that 
		can feed into csvmidi '''

		verbose("Getting csv string for Piece object")

		s = ""
		s += self.arrToCsv(self.head) + '\n'
		for track in self.tracks:
			endRow = ""

			# first print out all the headers
			for row in track.headers:

				# leave the "End_track" header for the end
				if row[TYPE].strip() == "End_track":
					endRow = row
				# This should only print if something broke
				elif row[TYPE].strip() == "End_of_file":
					print "THE END OF FILE IS IN TRACK:", track.getTrackNum()
				else:
					s += self.arrToCsv(row) + '\n'

			# merge the notes and non-note channel events
			noteCount = 0
			evCount = 0
			while noteCount < len(track.notes) or evCount < len(track.channelEvents):
				# We are out of notes to print
				if noteCount >= len(track.notes):
					s += self.arrToCsv(track.channelEvents[evCount]) + '\n'
					evCount += 1
				# the next note is before the next event, or we are out of events
				elif (evCount >= len(track.channelEvents) or
					(track.notes[noteCount].time <= int(track.channelEvents[evCount][TIME]))):
					s += track.notes[noteCount].toStr() + '\n'
					noteCount += 1
				# if the next event is before the next note
				else:
					s += self.arrToCsv(track.channelEvents[evCount]) + '\n'
					evCount += 1
			s += self.arrToCsv(endRow) + '\n'
		s += self.arrToCsv(self.endFile) + '\n'
		return s

	def csvPrint(self):
		print self.arrToCsv(self.head)
		for track in self.tracks:
			endRow = ""

			# first print out all the headers
			for row in track.headers:

				# leave the "End_track" header for the end
				if row[TYPE].strip() == "End_track":
					endRow = row
				# This should only print if something broke
				elif row[TYPE].strip() == "End_of_file":
					print "THE END OF FILE IS IN TRACK:", track.getTrackNum()
				else:
					print self.arrToCsv(row)

			# merge the notes and non-note channel events
			noteCount = 0
			evCount = 0
			while noteCount < len(track.notes) or evCount < len(track.channelEvents):
				# We are out of notes to print
				if noteCount >= len(track.notes):
					print self.arrToCsv(track.channelEvents[evCount])
					evCount += 1
				# the next note is before the next event, or we are out of events
				elif (evCount >= len(track.channelEvents) or
					(track.notes[noteCount].time <= int(track.channelEvents[evCount][TIME]))):
					print track.notes[noteCount].toStr()
					noteCount += 1
				# if the next event is before the next note
				else:
					print self.arrToCsv(track.channelEvents[evCount])
					evCount += 1
			print self.arrToCsv(endRow)
		print self.arrToCsv(self.endFile)



def fromHorizontal(notes):
	''' This goes from the horizontal representation to a piece object '''

	verbose("Converting Horizontal form to Piece Object")

	newP = Piece()
	newP.head = START_HEADERS_LIST
	newP.endFile = END_HEADERS_LIST
	numLines = len(notes[0])
	tracks = []
	prevNote = -1
	time = 0

	# deal with first note case

	for li in range(1,numLines+1): # line/track number is one-indexed
		line = genStandardHeadersList(li)
		endTime = 0
		for timeStep in range(len(notes)):
			time = timeStep*MIDI_CONVERSION # about one 16th note happens every 4 time units in midi
			t = notes[timeStep][li-1] # li-1 compensates for 1-indexing
			# If this is the first note, just append it
			if timeStep == 0:
				if t != 0: # do nothing if line starts with a rest
					line.append(noteWithPitch(t,li,time,True).toArr())
			# otherwise, only do something if the note has changed
			elif t != prevNote:
				line.append(noteWithPitch(prevNote, li, time-MIDI_SWITCH, False).toArr())
				if t != 0: # don't make next note play if it's a rest
					line.append(noteWithPitch(t,li,time,True).toArr())
			# set prevNote for next round to current note
			prevNote = t
			endTime = time

		# append the end of track line
		line.append(genEndHeadersList(li, endTime))
		# create a track from this note array
		track = Track(line)

		tracks.append(track)
	newP.tracks = tracks

	return newP



def oneHotToHorizontal(notes):
	''' Takes a 3 dimensional note array of the oneHotHorizontal form and returns
	the horizontal form'''

	verbose("Converting OneHotHorizontal form to Horizontal form")

	newArr = []
	for time in notes:
		newTime = []
		for line in time:
			if 1 in line[1:]:
				pitch = line[0]*12 + line[1:].index(1)
			else:
				pitch = 0
			newTime.append(pitch)
		newArr.append(newTime)

	return newArr



def fromOneHotHorizontal(notes):
	''' takes in an array of OneHotHorizontal vectors, and makes a Piece object
	arr is a 3 dimensional array, arr[timeStep][line][index in pitch vector]'''

	verbose("Converting OneHotHorizontal form to Piece object")

	horizNotes = oneHotToHorizontal(notes)
	return fromHorizontal(horizNotes)


def twoHotToHorizontal(notes):
	''' Takes a piece in twoHotHorizontal representation and transforms it into
	Horizontal representation '''

	verbose("Converting TwoHotHorizontal form to Horizontal form")
	

	oneHot = []
	# convert to one hot
	for time in notes:
		newTime = []
		for line in time:
			#print "two hot input", line
			# if the pitch vector is all 0s, then it is a rest
			if not(1 in line[8:]): 
				newTime.append([1] +[0]*12)
				#print "REST"
			else:
				octave = line[0:8].index(1)
				newTime.append([octave] + line[8:])
				#print "two hot output", [octave] + line[8:]

		oneHot.append(newTime)

	return oneHotToHorizontal(oneHot)

def fromTwoHotHorizontal(notes):
	''' takes a piece in TwoHotHorizontal and makes a piece objects from it
	it returns the piece object '''

	verbose("Converting TwoHotHorizontal form to Piece object")


	horizNotes = twoHotToHorizontal(notes)
	return fromHorizontal(horizNotes)


def CCToHorizontal(notes):
	''' Takes a piece in CC representation and converts it to horizontal '''

	verbose("Converting TwoHotHorizontal form to Horizontal form")
	
	CC = []
	# convert to one hot
	for time in notes:
		#print "time", time
		newTime = []
		for line in time:
			# if the pitch vector is all 0s, then it is a rest
			if not(1 in line[8:]): 
				#print "REST"
				newTime.append([1]*8 +[0]*12)
			else:
				pitch = 0 # default
				# These statements account for the possibility that all the bits
				# are 1s, in which case the last bit should be 1
				if not(0 in line[8:]):
					pitch = 11
				else:
					pitch = line[8:].index(0) - 1
				newPitchVec = [0]*12
				newPitchVec[pitch] = 1

				octave = 0 # default
				if not(0 in line[:8]):
					octave = 7
				else:
					octave = line[0:8].index(0) - 1
				newOctaveVec = [0]*8
				newOctaveVec[octave] = 1
				#print "newVec", newOctaveVec + newPitchVec
				newTime.append(newOctaveVec + newPitchVec)

		CC.append(newTime)

	return twoHotToHorizontal(CC)

def fromCC(notes):
	''' takes a piece in CC and makes a piece objects from it
	it returns the piece object '''

	verbose("Converting CC form to Piece object")

	horizNotes = CCToHorizontal(notes)
	return fromHorizontal(horizNotes)







def verbose(str):
	if False:
		print str





def main():
	''' This just has some random testing command right now '''


	filename = 'ArtOfFugueExpo.csv'
	p = Piece()
	p.fromCSV(filename)
	print "horizontal", p.getHorizontal()

	f = open('bachback.csv', 'w')

	
	#print "\n\n\nCC", p.getCC()[32:35]
	backP = fromCC(p.getCC())
	f.write(backP.csv())
	
	
	#f.write(p.csv())
	f.close()
	print p.head
	print p.endFile


if __name__ == '__main__':
	main()



