import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft
import wave
import numpy as np

from scipy.io import wavfile
import noisereduce as nr
from pyAudioAnalysis import audioBasicIO as aIO

class draggable_lines:
    def __init__(self, ax, id, kind, XorY, minY, maxY, color):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.o = kind
        self.XorY = XorY
        self.maxY = maxY
        self.id_num = id


        if kind == "h":
            x = [-1, 1]
            y = [XorY, XorY]

        elif kind == "v":
            x = [XorY, XorY]
            y = [minY, maxY]
        self.line = lines.Line2D(x, y, picker=2, linestyle='--',color = color)
        self.id_label = ax.annotate(id, (x[0], maxY))
        self.ax.add_line(self.line)
        self.c.draw_idle()
        self.sid = self.c.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            print("line selected ", event.artist)
            self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self.c.mpl_connect("button_press_event", self.releaseonclick)

            try:
                self.id_label.remove()
            except:
                print("Caught annotate remove error")

    def followmouse(self, event):
        if self.o == "h":
            self.line.set_ydata([event.ydata, event.ydata])
        else:
            self.line.set_xdata([event.xdata, event.xdata])


        self.c.draw_idle()

    def releaseonclick(self, event):
        if self.o == "h":
            self.XorY = self.line.get_ydata()[0]
        else:
            self.XorY = self.line.get_xdata()[0]



        print (self.XorY)

        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)


        self.id_label = ax.annotate(self.id_num, (self.line.get_xdata()[0], self.maxY))




def read_audio_file(file_directory):
    wav_obj = wave.open(file_directory, 'rb')
    rate, data = wavfile.read(file_directory)
    # wav_obj = wave.open(audio_file_name, 'rb')
    # Getting the sampling rate
    sample_freq = wav_obj.getframerate()
    # Getting Number of individual frames
    n_samples = wav_obj.getnframes()
    # Getting duration of audio file
    t_audio = n_samples / sample_freq
    # Getting number of channels of sound
    n_channels = wav_obj.getnchannels()
    # Getting amplitude of wave at each frame
    signal_wave = wav_obj.readframes(n_samples)
    # To get signal values from this, we have to turn to numpy

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    # Before we get to plotting signal values, we need to calculate the time at which each sample is taken.
    # This is simply the total length of the track in seconds, divided by the number of samples
    #  We can use linspace() from numpy to create an array of timestamps

    times = np.linspace(0, n_samples / sample_freq, num=n_samples)

    return [data, rate, signal_array, times, t_audio]

# Function to read Sentences.txt and return all sentences in a list
def get_sentences(filename):
    with open(filename, encoding='utf8') as file:
        lines = [line.rstrip() for line in file]

    return lines

# Function to return string with spaces replaced with underscore
def remove_space(str):
    a = str
    a1 = ""
    for i in range(len(a)):
        if a[i] == ' ':
            a1 = a1 + '_'
        else:
            a1 = a1 + a[i]
    return a1


if __name__ == "__main__":

    roman_sentences = get_sentences("../Dictionary/roman_urdu_sentences.txt")
    urdu_sentences = get_sentences("../Dictionary/urdu_sentences.txt")

    speaker_id = "14"

    # Iterating over directory of a single speaker
    for i in range(0,108):

        #print(roman_sentences[i])
        #print(urdu_sentences[i])

        curr_sent_r = roman_sentences[i] # current sentence in roman urdu
        curr_sent_u = urdu_sentences[i]  # current sentence in urdu

        #curr_sent_r="aap kidhar thay nai kyun chhae"
        audio_file_dir = "../" + "Dataset/" + "Urdu/" + speaker_id + "/" + remove_space(curr_sent_r) + "/" +   '_audio.wav'
        print(audio_file_dir)

        # load data
        [data, rate, signal_array, time_array, t_audio] = read_audio_file(audio_file_dir)
    
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        signal_array = np.frombuffer(data, dtype=np.int16)
        reduce_signal_array = np.frombuffer(reduced_noise, dtype=np.int16)
    
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(time_array, reduce_signal_array)
    
        # Making vertical lines on the plot
    
        line1 = draggable_lines(ax, "START", "v", 0, min(reduced_noise), max(reduced_noise),'red')
        line2 = draggable_lines(ax, 2, "v", 0.4, min(reduced_noise), max(reduced_noise),'red')
        line3 = draggable_lines(ax, 3, "v", 0.8, min(reduced_noise), max(reduced_noise),'red')
        line4 = draggable_lines(ax, 4, "v", 1.2, min(reduced_noise), max(reduced_noise),'red')
        line5 = draggable_lines(ax, 5, "v", 1.6, min(reduced_noise), max(reduced_noise),'red')
        line6 = draggable_lines(ax, 6, "v", 2, min(reduced_noise), max(reduced_noise),'red')
        line7 = draggable_lines(ax, 7, "v", 2.4, min(reduced_noise), max(reduced_noise),'red')
        line8 = draggable_lines(ax, 8, "v", 2.8, min(reduced_noise), max(reduced_noise),'red')
        line9 = draggable_lines(ax, "END", "v", max(time_array), min(reduced_noise), max(reduced_noise),'red')
    
    
        ax.set_title(curr_sent_r )
        ax.set_ylabel('Signal Value')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, t_audio)
        plt.show()
        print(ax)
    
        print("\n \n \n \n")

        urdu_toks = urdu_sentences[i].split()
        rom_toks = roman_sentences[i].split()


        print( "<سل> " ,float(f'{min(time_array):.4f}'),"   ",float(f'{line2.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[0] , "  ", float(f'{line2.line.get_xdata()[0]:.4f}'),"   ",float(f'{line3.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[1] , "  ",float(f'{line3.line.get_xdata()[0]:.4f}'),"   ",float(f'{line4.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[2] , "  ",float(f'{line4.line.get_xdata()[0]:.4f}'),"   ",float(f'{line5.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[3] , "  ",float(f'{line5.line.get_xdata()[0]:.4f}'),"   ",float(f'{line6.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[4] , "  ",float(f'{line6.line.get_xdata()[0]:.4f}'),"   ",float(f'{line7.line.get_xdata()[0]:.4f}'))
        print( urdu_toks[5] , "  ",float(f'{line7.line.get_xdata()[0]:.4f}'),"   ",float(f'{line8.line.get_xdata()[0]:.4f}'))
        print( "<سل> " , float(f'{line8.line.get_xdata()[0]:.4f}'),"   ",float(f'{max(time_array):.4f}'))

        align_text=""
        align_text +="س " +str(float(f'{min(time_array):.4f}'))+"   "+str(float(f'{line2.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[0] + "  "+ str(float(f'{line2.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line3.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[1] + "  "+str(float(f'{line3.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line4.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[2] + "  "+str(float(f'{line4.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line5.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[3] + "  "+str(float(f'{line5.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line6.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[4] + "  "+str(float(f'{line6.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line7.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +=urdu_toks[5] + "  "+str(float(f'{line7.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{line8.line.get_xdata()[0]:.4f}'))+'\n'
        align_text +="س " + str(float(f'{line8.line.get_xdata()[0]:.4f}'))+"   "+str(float(f'{max(time_array):.4f}'))

        #new_dir = "./" + str(speaker_id) + "/" + remove_space(curr_sent_r)
        #new_dir =  "../" + "Dataset/" + "Urdu/" + speaker_id + "/" + remove_space(curr_sent_r) + "/" + 'align'
        new_file_name = "../" + "Dataset/" + "Urdu/" + speaker_id + "/" + remove_space(curr_sent_r) + "/" + 'align'
        text_file = open(new_file_name, "w",encoding="utf-8")

        n = text_file.write(align_text)
        text_file.close()
