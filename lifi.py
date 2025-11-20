import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
fs = 360
duration = 0.5
quant_bits = 8
N = 64
cp = 16
M = 64
bits_per_sym = 6

# =========================
# ECG generate
# =========================
t = np.arange(0, duration, 1/fs)
ecg = 0.6*np.sin(2*np.pi*1.2*t) + 0.4*np.sin(2*np.pi*3*t) + 0.1*np.sin(2*np.pi*7*t)
ecg = (ecg - ecg.min())/(ecg.max()-ecg.min())
ecg = ecg.reshape(-1)
num_samples = len(ecg)

# =========================
# ADC 8bit
# =========================
levels = 2**quant_bits
data = np.round(ecg*(levels-1)).astype(int)
bits = ((data[:,None] & (1<<np.arange(quant_bits-1,-1,-1)))>0).astype(int).reshape(-1)

# =========================
# 64-QAM mapping
# =========================
def gray(x): return x ^ (x>>1)
lvl = np.array([-7,-5,-3,-1,1,3,5,7])
mapping = np.zeros(M,dtype=complex)
for v in range(M):
    I = lvl[gray((v>>3)&7)]
    Q = lvl[gray(v&7)]
    mapping[v] = I + 1j*Q
mapping = mapping/np.sqrt(np.mean(np.abs(mapping)**2))

pad = (-len(bits))%bits_per_sym
if pad: bits = np.concatenate([bits,np.zeros(pad,int)])
sbits = bits.reshape(-1,bits_per_sym)
ints = sbits.dot(1<<np.arange(bits_per_sym-1,-1,-1))
qam = mapping[ints]

# =========================
# OFDM TX
# =========================
numOFDM = int(np.ceil(len(qam)/N))
pad2 = numOFDM*N - len(qam)
if pad2: qam = np.concatenate([qam,np.zeros(pad2,complex)])
X = qam.reshape(N,numOFDM)
ifft_data = np.fft.ifft(X,axis=0)
ofdm_tx = np.vstack([ifft_data[-cp:],ifft_data]).reshape(-1)   # THIS goes to OSTBC

# =========================
# OSTBC ENCODER
# =========================
ofdm = ofdm_tx
L = len(ofdm)
if L%2!=0:
    ofdm = np.concatenate([ofdm, np.array([0+0j])])
tx1=[]
tx2=[]
for i in range(0,len(ofdm),2):
    s0=ofdm[i]
    s1=ofdm[i+1]
    tx1.append(s0)
    tx1.append(-np.conj(s1))
    tx2.append(s1)
    tx2.append(np.conj(s0))
tx1=np.array(tx1)
tx2=np.array(tx2)

# =========================
# CHANNEL
# =========================
h=(np.random.randn(2)+1j*np.random.randn(2))/np.sqrt(2)
SNR_dB=35
SNR_lin=10**(SNR_dB/10)
sig=h[0]*tx1+h[1]*tx2
P=np.mean(np.abs(sig)**2)
sigma=np.sqrt(P/SNR_lin)
noise=sigma*(np.random.randn(len(sig))+1j*np.random.randn(len(sig)))
rx=sig+noise

# =========================
# OSTBC DECODER
# =========================
decoded=[]
den=np.abs(h[0])**2+np.abs(h[1])**2
for i in range(0,len(rx),2):
    y1=rx[i];y2=rx[i+1]
    r0=np.conj(h[0])*y1 + h[1]*np.conj(y2)
    r1=np.conj(h[1])*y1 - h[0]*np.conj(y2)
    decoded.append(r0/den)
    decoded.append(r1/den)
decoded = np.array(decoded)[:len(ofdm_tx)]

# =========================
# PLOT OSTBC BLOCK
# =========================
plt.figure(figsize=(10,8))
plt.subplot(3,1,1); plt.plot(ofdm_tx.real);         plt.title("OFDM samples (input to OSTBC)"); plt.grid(True)
plt.subplot(3,1,2); plt.plot(tx1.real);plt.plot(tx2.real); plt.title("TX1 & TX2 after OSTBC Encoding"); plt.grid(True)
plt.subplot(3,1,3); plt.plot(decoded.real);         plt.title("After OSTBC Decoding"); plt.grid(True)
plt.tight_layout()
plt.show()

samples_per_ofdm=N+cp
total=samples_per_ofdm*numOFDM
decoded=decoded[:total]
rxmat=decoded.reshape(samples_per_ofdm,numOFDM)
rx_no=rxmat[cp:,:]
rx_fft=np.fft.fft(rx_no,axis=0)
rx_sy=rx_fft.reshape(-1)[:len(qam)]

# =========================
# QAM DEMOD
# =========================
rx_int=[]
for s in rx_sy:
    d=np.abs(mapping-s)
    rx_int.append(int(np.argmin(d)))
rx_int=np.array(rx_int)
rx_bits=((rx_int[:,None]&(1<<np.arange(bits_per_sym-1,-1,-1)))>0).astype(int).reshape(-1)
rx_bits=rx_bits[:len(bits)]
pad3=(-len(rx_bits))%quant_bits
if pad3: rx_bits=np.concatenate([rx_bits,np.zeros(pad3,int)])
recon=rx_bits.reshape(-1,quant_bits).dot(1<<np.arange(quant_bits-1,-1,-1))
ecg_rx=(recon.astype(float)/(levels-1))[:num_samples]

# =========================
# ECG COMPARE
# =========================
plt.figure(figsize=(10,5))
plt.plot(ecg,label='Tx ECG')
plt.plot(ecg_rx,label='Rx ECG')
plt.title("ECG comparison after FULL chain")
plt.legend();plt.grid(True)
plt.show()
