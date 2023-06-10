from moviepy.editor import VideoFileClip
from pymediainfo import MediaInfo

# Function to read aligns of a video file (reads Urdu words)
def read_align(file_directory):

    file_directory = file_directory + "\\align"

    # Op
    file1 = open(file_directory, 'r', encoding='utf8')

    lines = file1.readlines()

    aligns = []

    for ln in lines:

        toks = ln.split()

        aligns.append(toks)

    return aligns

# Function to read aligns of a video file (FOR GRID-CORPUS)
def read_align_eng(file_directory):



    media_info = MediaInfo.parse(file_directory)
    # duration in seconds
    duration = media_info.tracks[0].duration / 1000

    # Parsing align directory
    tmp_dir = file_directory
    tmp_dir = tmp_dir.split('\\')
    align_dir = tmp_dir[0] + "\\" + tmp_dir[1] + "\\" + tmp_dir[2] + "\\" + tmp_dir[3] + "\\" + "align" + "\\" + tmp_dir[4].replace(".mpg",".align")

    # Opening and reading align file
    file1 = open(align_dir, 'r')

    lines = file1.readlines()

    aligns = []

    # Variables to get start and end of each segment
    total_fr = 74500

    for ln in lines:
        toks = ln.split()

        # Calculating start and end time of each word
        start_time = (float(toks[0])/total_fr) * duration
        end_time = (float(toks[1])/total_fr) * duration

        # appending in list
        aligns.append([toks[2], round(start_time, 2), round(end_time, 2)])

    return aligns



# Read_align() tester
def main1():
    file = "..\\Dataset\\Urdu\\3\\aap_kidhar_hai_nai_kab_theen"

    align = read_align(file)

    print("IND 0 : ",align[0][0] )
    print("IND 1 : ",align[0][1] )
    print("IND 2 : ",align[0][2] )

    for i in align:
        print(i[0], i[1], i[2])

# Read_align_GC() tester
def main2():

    file = "..\\Dataset\\GC\\s1\\bbaf2n.mpg"


    align = read_align_eng(file)


    print("IND 0 : ",align[0][0] )
    print("IND 1 : ",align[0][1] )
    print("IND 2 : ",align[0][2] )

    for i in align:
        print(i[0], i[1], i[2])

    print(len(align))

# Testing read function
if __name__ == "__main__":



    main2()







