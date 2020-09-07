import models
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

model = models.NBeatsNet()

model.load_weights('C:\\Users\\HERO\\PycharmProjects\\Kist_model1\\saved_model\\25.hdf5')



def complexSignal(f1, f2, a1, a2, data_points=481, dT=1/240.0, noisy=False,
                  mean=0, std=10, separate_signals=True):
    if noisy:
        noise = np.random.normal(mean, std, size=data_points)
    else:
        noise = np.zeros(shape=(data_points))

    tremor1 = []
    tremor2 = []

    frequencies = np.zeros(shape=(2, data_points))

    t = 0
    ran1 = np.random.random()
    ran2 = np.random.random()
    for i in range(data_points):
        t += dT
        tremor1.append(a1 * math.sin(2 * math.pi * f1 * t + ran1*2*math.pi))
        tremor2.append(a2 * math.cos(2 * math.pi * f2 * t + ran2*2*math.pi))
        if a1 > a2:
            frequencies[0][i] = f1
            frequencies[1][i] = f2
        else:
            frequencies[0][i] = f2
            frequencies[1][i] = f1
    if separate_signals:
        return np.array(tremor1), np.array(tremor2), noise
    else:
        return np.array(tremor1) + np.array(tremor2) + noise, frequencies


freq1 = 0.140842365
freq2 = 9.720146623

amp1 = 2.3464099224
amp2 = 0.0115641271

test_data_voluntary,test_data_tremor, _ = complexSignal(freq1,freq2, amp1, amp2, data_points= 6000)

test_data_voluntary,test_data_tremor  = test_data_voluntary, test_data_tremor
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)

test_data_x,_ = univariate_data(test_data_voluntary+test_data_tremor, 0, None, 480, 0 )
_, test_data_y = univariate_data(test_data_voluntary, 0, None, 480, 0 )

plt.plot(test_data_voluntary[480:][:100])
plt.plot((test_data_voluntary+test_data_tremor)[480:][:100])
plt.plot(model.predict(test_data_x).reshape(-1,)[:100])
plt.xticks([x for x in range(100) if x%30==0], ['{:2}'.format(float(x/240.0)) for x in range(100) if x%30==0])
plt.show()

plt.plot(test_data_voluntary[480:])
plt.plot((test_data_voluntary+test_data_tremor)[480:])
plt.plot(model.predict(test_data_x).reshape(-1,))
plt.xticks([x for x in range(len(test_data_tremor)) if x%480==0], ['{:2}'.format(float(x/240.0)) for x in range(len(test_data_tremor)) if x%480==0])
plt.show()


def custom_loss(y_true, y_pred):
    a =model.predict(test_data_x).reshape(-1, )
    return tf.reduce_mean(((a[:-1]-a[1:])**2)>0.025)


# Calculate FFT ....................
Fs = 240                    # Sampling frequency
T = 25/6                     # Sample interval time
te= 25/6*len(test_data_x)            # End of time
t = [x*25/6 for x in range(len(test_data_x))]   # Time vector

n=len(test_data_x)        # Length of signal
NFFT=n      # ?? NFFT=2^nextpow2(length(y))  ??
k=np.arange(NFFT)
f0=k*Fs/NFFT    # double sides frequency range
f0=f0[range(math.trunc(NFFT/2))]        # single sied frequency range

Y=np.fft.fft(model.predict(test_data_x).reshape(-1,))/NFFT        # fft computing and normaliation
Y=Y[range(math.trunc(NFFT/2))]          # single sied frequency range
amplitude_Hz = 2*abs(Y)
phase_ang = np.angle(Y)*180/np.pi



# figure 1 ..................................
plt.figure(num=2,dpi=100,facecolor='white')
plt.subplots_adjust(hspace = 1.8, wspace = 0.3)
plt.subplot(3,1,1)

plt.plot(t,model.predict(test_data_x).reshape(-1),'r')
plt.title('Signal FFT analysis')
plt.xlabel('time($ms$)')
plt.ylabel('y')
plt.xticks([x for x in range(10000) if x%10000==0], [int(x/1000) for x in range(10000) if x%10000==0]) ### 쑤파
#plt.xlim( 0, 0.1)

# Amplitude ....
#plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(3,1,2)

# Plot single-sided amplitude spectrum.

plt.plot(f0,amplitude_Hz,'r')   #  2* ???
plt.xticks(np.arange(0,20,1))
plt.xlim( 0, 20)
plt.ylim( 0, 0.05)
#plt.title('Single-Sided Amplitude Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('amplitude')
plt.grid()

# Phase ....
#plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(3,1,3)
plt.plot(f0,phase_ang,'r')   #  2* ???
plt.xlim( 0, 5)
plt.ylim( -250, 250)
#plt.title('Single-Sided Phase Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('phase($deg.$)')
plt.xticks([0, 5])
plt.yticks([-180, -90, 0, 90, 180])
plt.grid()

plt.savefig("./test_figure2.png",dpi=300)
plt.show()



# Calculate FFT ....................
Fs = 240                    # Sampling frequency
T = 25/6                     # Sample interval time
te= 25/6*len(test_data_x)            # End of time
t = [x*25/6 for x in range(len(test_data_x))]   # Time vector

n=len(test_data_x)        # Length of signal
NFFT=n      # ?? NFFT=2^nextpow2(length(y))  ??
k=np.arange(NFFT)
f0=k*Fs/NFFT    # double sides frequency range
f0=f0[range(math.trunc(NFFT/2))]        # single sied frequency range

Y=np.fft.fft((test_data_voluntary+test_data_tremor)[480:].reshape(-1,))/NFFT        # fft computing and normaliation
Y=Y[range(math.trunc(NFFT/2))]          # single sied frequency range
amplitude_Hz = 2*abs(Y)
phase_ang = np.angle(Y)*180/np.pi



# figure 1 ..................................
plt.figure(num=2,dpi=100,facecolor='white')
plt.subplots_adjust(hspace = 1.8, wspace = 0.3)
plt.subplot(3,1,1)

plt.plot(t,(test_data_voluntary+test_data_tremor)[480:],'r')
plt.title('Signal FFT analysis')
plt.xlabel('time($ms$)')
plt.ylabel('y')
plt.xticks([x for x in range(n) if x%1200==0], [int(x/240) for x in range(n) if x%1200==0])
#plt.xlim( 0, 0.1)

# Amplitude ....
#plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(3,1,2)

# Plot single-sided amplitude spectrum.

plt.plot(f0,amplitude_Hz,'r')   #  2* ???
plt.xticks(np.arange(0,20,1))
plt.xlim( 0, 20)
plt.ylim( 0, 0.05)
#plt.title('Single-Sided Amplitude Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('amplitude')
plt.grid()

# Phase ....
#plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(3,1,3)
plt.plot(f0,phase_ang,'r')   #  2* ???
plt.xlim( 0, 5)
plt.ylim( -250, 250)
#plt.title('Single-Sided Phase Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('phase($deg.$)')
plt.xticks([0, 5])
plt.yticks([-180, -90, 0, 90, 180])
plt.grid()

plt.savefig("./test_figure2.png",dpi=300)
plt.show()