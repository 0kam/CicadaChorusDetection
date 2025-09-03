import pyaudio
from queue import Queue
import numpy as np
import time

class NoMicrophoneFoundError(Exception):
    pass

class AudioDevice:
    """The base class of all AudioDevice classes.
    All audio device classes must inherit this class.
    An AudioDevice instance performs non-blocking streaming of raw audio data. 
    The sampling rate will be automatically searched.

    Attributes:
        channels (int): The channel number.
        chunk_sec (float): The chunk length (in seconds).
        mic_id (int): The selected mic's ID.
        mic_name (str): The selected mic's name.
        sampling_rate (float): The sampling rate which depends on the mic.
        chunk_size (int): chunk_sec * sampling_rate
        q (queue.Queue): A Queue object for audio data exchanging. 
        stream: (pyaudio stream): The audio streaming.
    """
    def __init__(
        self,
        channels : int,
        chunk_sec : float,
        max_delay = 0.1
        ) -> None:
        """
        
        """
        self.channels = channels
        self.chunk_sec = chunk_sec
        self.max_delay = max_delay
        self.q = Queue()
        self.time = Queue()
        self.pa = pyaudio.PyAudio()
        mic_id, default_sampling_rate = self._mic_id()
        self.mic_id = mic_id
        self.sampling_rate = int(default_sampling_rate)
        self.chunk_size = int(self.chunk_sec * self.sampling_rate)
        self.stream = self.pa.open(
                            format = pyaudio.paInt16,
                            channels = self.channels,
                            rate=self.sampling_rate,
                            input=True,
                            frames_per_buffer=self.chunk_size,
                            input_device_index=mic_id,
                            stream_callback=self.callback
                            )
    
    def callback(self, in_data, frame_count, time_info, status):
        in_data = np.frombuffer(in_data, dtype=np.int16)
        self.time.put(time.time())
        self.q.put(in_data)
        return None, pyaudio.paContinue
    
    def start_stream(self):
        self.stream.start_stream()
    
    def stop_stream(self):
        self.stream.stop_stream()
    
    def get(self, normalize = False):
        while (time.time() - self.time.get()) > self.max_delay:
            self.q.get()
        a = self.q.get()
        if normalize:
            a = -1 + a / (a.max()-a.min()) * 2
        return a
    
    def _mic_id(self, keyword):
        """
        Get microphone id
        Microphones are searched by the given keyword.
        Returns
        -------
        ids
            index of the USB microphone.
        Raises
        ------
        NoMicrophoneFoundError
            No microphone was found.
        """
        pa = pyaudio.PyAudio()
        id = None
        self.mic_name = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(info["name"])
            if keyword in info["name"]:
                id = i
                default_sampling_rate = info["defaultSampleRate"]
                self.mic_name = keyword
                print("Found {} Microphone {} with device index {}".format(keyword, info["name"], i))
                print(info)

        if id == None:
            raise NoMicrophoneFoundError("No microphone was found.")
        
        return id, default_sampling_rate
    
    def __del__(self):
        self.stop_stream()
        self.stream.close()
        self.pa.terminate()

class CustomMic(AudioDevice):
    """
    Using this class, you can select any mic searching with key words.
    This class treats a mic as monaural (whatever the chunnel size is) and simply returns raw audio data. 
    """
    def __init__(self, chunk_sec: float, keyword: str) -> None:
        self.mic_name = keyword
        super().__init__(channels=1, chunk_sec=chunk_sec)
    
    def _mic_id(self):
        return super()._mic_id(self.mic_name)