class SoundPreProcess():
    SR = 16000
    N_FFT = 1024
    HOP   = 256               
    N_MELS = 64               
    TOP_DB = 70               
    FMIN_MEL, FMAX_MEL = 50.0, 8000.0
    FMIN_F0,  FMAX_F0  = 50.0, 800.0     
    TARGET = 224