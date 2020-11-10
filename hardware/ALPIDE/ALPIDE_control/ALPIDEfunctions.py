import uhal
import time
from subprocess import call
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from scipy.optimize import curve_fit
from scipy import special

matplotlib.rcParams['text.usetex'] = True

#routine list (definitions)
power_off = 0b0000
power_on = 0b0001
initialize = 0b0010
read_register = 0b0100
read_out = 0b0101
write_cmd = 0b1000
write_register = 0b1001
trigger_cmd = 0b1010
clear_err = 0b1111

C_inj=230e-18	#capacitance for analog pulse [Farad]


#defining error function with parameter mu, sigma
def fit_err_func(x, mu, sigma):
	return 0.5*(1+special.erf((x-mu)/(math.sqrt(2)*sigma)))

#defining linear function with parameter a,b
def linear_func(x, a, b):
	return a+b*x

def send_cmd(cmd):	#send command to perform into FPGA
	hw.getNode("CSR.ctrl.op_sw").write(cmd)	#change the value of op_sw
	hw.getNode("CSR.ctrl.strt").write(0b1)	#start variable changed to '1'
	#hw.dispatch()
	#time.sleep(0.001)
	hw.getNode("CSR.ctrl.strt").write(0b0)	#start variable changed back to '0'
	hw.dispatch()

#initialization routine, set the xml file and the device
def init():
	global manager
	global hw
	manager = uhal.ConnectionManager("file://ALPIDE_connection.xml")
	hw = manager.getDevice("ALPIDE")
	hw.getNode("CSR.ctrl.strt").write(0b0)
	hw.dispatch()
	print 'Initialization comlpete'

def get_status():	#get all status of ALPIDE CARRIER
	powered = hw.getNode("CSR.status.powered").read()
	initialized = hw.getNode("CSR.status.init").read()
	busy = hw.getNode("CSR.status.busy").read()
	FIFO_full = hw.getNode("CSR.status.FIFO_full").read()
	FIFO_empty = hw.getNode("CSR.status.FIFO_empty").read()
	FIFO_prg_full = hw.getNode("CSR.status.FIFO_prog_full").read()
	mem_readable = hw.getNode("CSR.status.mem_readable").read()
	err_slave = hw.getNode("CSR.status.err_slave").read()
	err_idle = hw.getNode("CSR.status.err_idle").read()
	err_read = hw.getNode("CSR.status.err_read").read()
	err_chip_id = hw.getNode("CSR.status.err_chip_id").read()
	lkd_AC = hw.getNode("CSR.status.lkd_AC").read()
	lkd_ipbus = hw.getNode("CSR.status.lkd_ipbus").read()
	hw.dispatch()
	return powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus

def writeregister(addr,data):	#Write ALPIDE register
	hw.getNode("cmd_addr.WR_addr").write(addr)
	hw.getNode("cmd_addr.WR_data").write(data)
	hw.dispatch()
	send_cmd(write_register)

def readregister(addr):	#Read ALPIDE register
	hw.getNode("cmd_addr.RR_addr").write(addr)
	send_cmd(read_register)
	reg_read = hw.getNode("DATA.reg_read").read()
	hw.dispatch()
	return reg_read

def send_broadcast(CMD):	#send broadcast command
	hw.getNode("cmd_addr.OP_command").write(CMD)
	hw.dispatch()
	send_cmd(write_cmd)

#threshold test (it's performed within 2-3 hrs)
def global_thr_test(N, Masked_pixels):	#N--> number of tests per pixels (defalut it's 50) Masked_pixels-->list of masked pixels
	#initialte some lists
	Q_inj=[]	#charge injected per pixel
	ff_array=[]	#pixels fraction that respond to PULSE (fitting purposes)
	ffstd_array=[]	#standard deviation of pixel fraction
	thr=[]	#threshold list
	el_noise=[]	#electron noise list
	#global tt_fraction_fired
	#tt_fraction_fired=np.zeros((512,1024,120))	#y,x,DAC value
	#global tt_std
	#tt_std=np.zeros((512,1024,120))	#y,x,DAC value
	tt_fraction_fired=[]
	#enablePulserAllPixels(False)	#enable pulse on all pixels
	#maskAllPixels(True)	#mask all pixels
	#maskAllPixels(False)	#unmask all pixels
	#enablePulserAllPixels(True)	#enable pulse on all pixels

	writeregister(0x0004, 0x0060)  # Analog pulse, automatic strobe after pulse command
	writeregister(0x0606, 0x0000)  # Set PULSEL to 0.37V (DAC offset 370mV)

	writeregister(0x0001, 0x020E)  # mode cotrol register configuration
	writeregister(0x0005, 0x0002)  # STROBE duration (n+1) clk clycles
	writeregister(0x0006, 0x0000)  # gap between 2 STROBES (n+1) clk clycles
	writeregister(0x0007, 0x0013)  # delay from PULSE to STROBE (n+1) clk cycles
	writeregister(0x0008, 0x0016)  # PULSE duration n clk cylces

	for k in range(160):	#variable charge (from 0 to 1600 electrons)
		if k == 80:
			print "Half read complete\n"
		if k == 120:
			print "0.75 read complete\n"
		if k == 144:
			print "0.90 read complete\n"
		pulse_hitmap = np.zeros((512,1024))	#initate an empty matrix 1024x512
		Q_inj.append(((k)*7.06e-3*C_inj)/(1.602e-19)) #append charge value  Q=(VPULSEH-VPULSEL)*C_inj/e (10 electrons per step)
		writeregister(0x0605,k)  # Set PULSEH to (k)*7.06+370 mV
		time.sleep(0.001)	#DACs need time to stabilize
		for i in range(N):	#N measure
			for j in range(16):	#row region selection
				maskAllPixels(True)	#mask all pixels
				enablePulserAllPixels(False)	#disable Pulser
				enablePulserRow_Region(j,0xFFFF,True)	#pulse selected 16 rows (16384 pixels)
				maskRow_Region(j,0xFFFF,False)	#unmask selected 16 rows (16384 pixels)
				send_broadcast(0x78)  #send PULSE command
				time.sleep(0.0002)
				maskAllPixels(True)	#mask all pixels
				enablePulserAllPixels(False)	#disable Pulser
				enablePulserRow_Region(j+16,0xFFFF,True)	#pulse selected 16 rows (16384 pixels)
				maskRow_Region(j+16,0xFFFF,False)	#unmask selected 16 rows (16384 pixels)
				send_broadcast(0x78)  #send PULSE command
				time.sleep(0.0002)
				pulse_hitmap = pulse_hitmap + read_PULSE() #cumulative PULSE hitmap
		tt_fraction_fired.append(pulse_hitmap/N)	#load the fraction N_fired/N_injected matrix
		pulse_hitmap = []	#clear the pulse matrix
	print "Fit processing..\n"
	for x in range(1024):
		for y in range(512):
			if [y,x] in Masked_pixels:
				print "Pixel [x=%d, y=%d] doensen't contribute to threshold as it is noisy\n" %(x, y)
			else:
				if (x == 511) and (y == 255):
					print "Half processing complete\n"
				if (x == 767) and (y == 255):
					print "0.75 processing complete\n"
				for k in range(160):
					if tt_fraction_fired[k][y,x]<=1:	#check if the fraction is less than 1, if not it will change it's value to 1 with sigma 0f 0.15
						ff_array.append(tt_fraction_fired[k][y,x])
						ffstd_array.append(math.sqrt(tt_fraction_fired[k][y,x]*(1-tt_fraction_fired[k][y,x])/N))	#sigma (bernoulli)
					else:
						ff_array.append(1)
						ffstd_array.append(0.15)
				fit_error = [max(i,0.0072) for i in ffstd_array]	#if the sigma it's 0 it will change it to an uniform distribution sigma
				if np.sum(ff_array) > 40:	#check if it's a dead pixel
					popt = curve_fit(fit_err_func, Q_inj, ff_array, p0=[200, 10], sigma=fit_error, absolute_sigma=True, bounds=([0, 0.3], [1600, 120]))	#fitting
					thr.append(popt[0])	#append the thresold value (mean of error function)
					el_noise.append(popt[1])	#append the electron noise value (sigma of error function)
				else:
					print "Pixel [x=%d, y=%d] is dead\n" %(x, y)
				ff_array=[]
				ffstd_array=[]

	np.save(os.path.join('Current_run_data/Threshold_test/Global_threshold_test', 'Thr'), thr)
	np.save(os.path.join('Current_run_data/Threshold_test/Global_threshold_test', 'Noise'), el_noise)

	#print mean and sigma of the distributions found
	print 'Threshold= %f +/- %f' % (np.mean(thr), np.std(thr))
	print 'Elecotrn noise= %f +/- %f' % (np.mean(el_noise), np.std(el_noise))

	#draw the distribustions histograms
	draw_histos(thr, el_noise)

#read one PULSE event and return a matrix with '0' and '1'
def read_PULSE():
	#initialization of some variables related to a given pixel's geographical coordinates
	region=0
	n_events=0
	n_idle=0
	trailer_detected=0
	empty_detected=0
	hitmap_matrix = np.zeros((512, 1024))	#ALPIDE matrix
	send_cmd(read_out)
	while (trailer_detected+empty_detected<2):	#read two data packets
		mem_readable = hw.getNode("CSR.status.mem_readable").read()	#wait until the data is ready to be read
		hw.dispatch()
		if mem_readable:	#decode the income packet
			FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
			hw.dispatch()
			for word in FIFO:
				if (word & 0xF00000) == 0xA00000:  # chip header
					plane_id = (word & 0x0F0000) >> 16
					bc = (word & 0x00FF00) >> 8
				elif (word & 0xF00000) == 0xB00000:  # chip trailer
					ro_flags = (word & 0x0F0000) >> 16
					trailer_detected=trailer_detected+1
				elif (word & 0xF00000) == 0xE00000:  # chip empty
					plane_id = (word & 0x0F0000) >> 16
					bc = (word & 0x00FF00) >> 8
					empty_detected=empty_detected+1
				elif (word & 0xE00000) == 0xC00000:  # reagion header
					region = (word >> 16) & 0x1F
				elif (word & 0xC00000) == 0x400000:  # short
					word_shifted = word >> 8
					x = region << 5 | word_shifted >> 9 & 0x1E | (
								word_shifted ^ word_shifted >> 1) & 0x1  # region*32 + encoderid*2 + addr correction
					y = word_shifted >> 1 & 0x1FF  # addr/2
					hitmap_matrix[y, x] = hitmap_matrix[y, x] + 1
					n_events = n_events + 1
				elif (word & 0xC00000) == 0x000000:  # long
					word_shifted = word >> 8
					address = word_shifted & 0x3ff
					x = region << 5 | word_shifted >> 9 & 0x1E | (word_shifted ^ word_shifted >> 1) & 0x1
					y = word_shifted >> 1 & 0x1FF
					hitmap_matrix[y, x] = hitmap_matrix[y, x] + 1
					hitmap = word & 0x7F
					n_events = n_events + 1
					for i in range(7):
						hitmap_shifted = hitmap >> i
						if (hitmap_shifted & 0x1) == 0x1:
							addressmap = address + i + 1
							xhm = region << 5 | word_shifted >> 9 & 0x1E | (
										addressmap ^ addressmap >> 1) & 0x1  # x hit map
							yhm = addressmap >> 1  # y hit map
							hitmap_matrix[yhm, xhm] = hitmap_matrix[yhm, xhm] + 1
							n_events = n_events + 1
				elif word == 0xFFFFFF:	#idle
					n_idle = n_idle + 1
			# elif word == 0xF1FFFF:	#busy on
			# elif word == 0xF0FFFF:	#busy off
			# else:
			hw.getNode("CSR.ctrl.mem_read").write(0b1)	#send the flag data read
			hw.getNode("CSR.ctrl.mem_read").write(0b0)
			hw.dispatch()
	hw.getNode("CSR.ctrl.ro_stop").write(0b1)	#stop read out
	hw.dispatch()
	print_error()	#print read error if present
	hw.getNode("CSR.ctrl.ro_stop").write(0b0)
	hw.dispatch()
	return hitmap_matrix

#drow thr and el_noise histograms
def draw_histos(thr_arr, el_noise_arr):
	fig, axes = plt.subplots(nrows=1, ncols=2)
	ax0, ax1 = axes.flatten()

	# the histogram of the data
	n0, bins0, patches0 = ax0.hist(thr_arr, bins='auto', density=1)
	n1, bins1, patches1 = ax1.hist(el_noise_arr, bins='auto', density=1)

	#Thr histo
	ax0.plot()
	ax0.set_xlabel('Thr[e$^{-}$]',fontsize='14')
	ax0.set_ylabel('a.u.',fontsize='14')
	ax0.set_title(r'Threshold',fontsize='16')
	ax0.grid(True)

	#El_noise histo
	ax1.plot()
	ax1.set_xlabel('Noise[e$^{-}$]',fontsize='14')
	ax1.set_ylabel('a.u.',fontsize='14')
	ax1.set_title(r'Noise',fontsize='16')
	plt.savefig("Threshold_scan", dpi=1000)

	fig.suptitle(r'Threshold scan (%d entries)' % len(thr_arr), fontsize='22')

	fig.tight_layout()
	plt.show()

def SP_thr_test(x,y):	#single pixel threshold test
	Q_inj_arr=[]
	N_fired_arr=[]
	N_fired_err_arr=[]
	file = open("Current_run_data/Threshold_test/SP_threshold_test/SP_Threshold_test.txt", "w")
	enablePulserAllPixels(False)	#enable pulse on all pixels
	maskAllPixels(True)	#mask all pixels
	maskPixel(x, y, False)	#unmask selected pixel
	enablePulserPixel(x, y, True)
	writeregister(0x0004, 0x0060)  # Analog pulse, automatic strobe after pulse command
	writeregister(0x0606, 0x0000)  # Set PULSEL to 370mV (DAC offset)
	for k in range (180):
		pulse_test_array = []  # initialize empty array
		Q_inj = ((k)*7.06e-3*C_inj)/(1.602e-19)
		writeregister(0x0605,k)  # Set PULSEH to k*7.06+370 mV
		time.sleep(0.0005)
		prf = thr_test() #pixel respond fraction
		mean = prf
		std = math.sqrt(prf*(1-prf)/40)
		file.write("%f	%f	0	%f\n" % (Q_inj, mean, std))
	file.close()
	writeregister(0x0004, 0x0000)
	maskAllPixels(False)	#unmask all pixels
	data = np.loadtxt('SP_Threshold_test.txt')
	charge = [row[0] for row in data]
	pixel_react_times  = [row[1] for row in data]
	yerror = [row[3] for row in data]
	fit_error = [max(i,0.0072) for i in yerror]
	popt, pcov = curve_fit(fit_err_func, charge, pixel_react_times, p0=[200, 10], sigma=fit_error, absolute_sigma=True,  bounds=([0, 1], [1800, 120]))
	print '''
Threshold=%f +/- %f
Electron noise=%f +/- %f
	''' % (popt[0],pcov[0,0],popt[1],pcov[1,1])

def thr_test():
	hitmap_matrix = np.zeros((512, 1024))
	region = 0
	n_events = 0
	n_idle = 0
	n_busy = 0
	n_data = 0
	writeregister(0x0001, 0x020D)  # mode cotrol register configuration (see above)
	writeregister(0x0005, 0x0002)  # STROBE duration 3+1 clk clycles
	writeregister(0x0006, 0x0000)  # gap between 2 STROBES
	writeregister(0x0007, 0x0013)  # delay from PULSE to STROBE 5+1 clk cycles
	writeregister(0x0008, 0x0016)  # PULSE duration 9+1 clk cylces
	for i in range(40):
		send_broadcast(0x78)  # send PULSE command
	send_cmd(read_out)
	for i in range(10):
		time.sleep(0.0002)
		mem_readable = hw.getNode("CSR.status.mem_readable").read()
		hw.dispatch()
		if mem_readable:
			FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
			hw.dispatch()
			n_events, n_idle, n_busy, n_data, region, hitmap_matrix = decode_block(FIFO, n_events, n_idle, n_busy, n_data, region, hitmap_matrix)
			hw.getNode("CSR.ctrl.mem_read").write(0b1)
			hw.getNode("CSR.ctrl.mem_read").write(0b0)
			hw.dispatch()
	hw.getNode("CSR.ctrl.ro_stop").write(0b1)
	hw.dispatch()
	#save_hitmap(hitmap_matrix, 'PulseHitmap.png')
	fired_fraction = float(n_events) / 40
	string_1 = 'Pixel x,y responds %f' % (fired_fraction * 100)
	string = string_1 + chr(37) + ' of the times\n'
	print string
	writeregister(0x0007, 0x0000)
	writeregister(0x0008, 0x0000)
	powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = get_status()
	if err_slave:  # more then 51 clk cycles driven by slave
		print
		"Slave drive on time out"
	if err_idle:
		print
		"high signal not detected on idle phase"
	if err_read:
		print
		"stop bit not detected"
	if err_chip_id:
		print
		"different chip ID recieved"
	time.sleep(0.001)
	hw.getNode("CSR.ctrl.ro_stop").write(0b0)
	hw.dispatch()
	return fired_fraction


def pulse_test(mode_cfg):
	hitmap_matrix = np.zeros((512, 1024))
	region = 0
	n_events = 0
	n_idle = 0
	n_busy = 0
	n_data = 0
	writeregister(0x0001, mode_cfg)  # mode cotrol register configuration (see above)
	writeregister(0x0005, 0x0002)  # STROBE duration 3 clk clycles
	writeregister(0x0006, 0x0000)  # gap between 2 STROBES
	writeregister(0x0007, 0x0008)  # delay from PULSE to STROBE 8+1 clk cycles
	writeregister(0x0008, 0x000E)  # PULSE duration 12 clk cylces
	send_broadcast(0x78)  # send PULSE command
	send_cmd(read_out)
	for i in range(800):
		time.sleep(0.00001)
		mem_readable = hw.getNode("CSR.status.mem_readable").read()
		hw.dispatch()
		if mem_readable:
			FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
			hw.dispatch()
			n_events, n_idle, n_busy, n_data, region, hitmap_matrix = decode_block(FIFO, n_events, n_idle, n_busy, n_data, region, hitmap_matrix)
			hw.getNode("CSR.ctrl.mem_read").write(0b1)
			hw.getNode("CSR.ctrl.mem_read").write(0b0)
			hw.dispatch()
	hw.getNode("CSR.ctrl.ro_stop").write(0b1)
	hw.dispatch()
	#save_hitmap(hitmap_matrix, 'PulseHitmap.png')
	pixel_fraction = float(n_events) / 200
	string_1 = 'Pulse test, %f' % (pixel_fraction * 100)
	string = string_1 + chr(37) + ' pixels respond\n'
	#print string
	writeregister(0x0007, 0x0000)
	writeregister(0x0008, 0x0000)
	powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = get_status()
	if err_slave:  # more then 51 clk cycles driven by slave
		print
		"Slave drive on time out"
	if err_idle:
		print
		"high signal not detected on idle phase"
	if err_read:
		print
		"stop bit not detected"
	if err_chip_id:
		print
		"different chip ID recieved"
	time.sleep(0.001)
	hw.getNode("CSR.ctrl.ro_stop").write(0b0)
	hw.dispatch()
	return pixel_fraction

#raw data block decodification
def decode_block(data_block, n_events, n_idle, n_busy, n_data, region, hitmap_matrix):	#data_block-->raw data
	for word in data_block:
		n_data = n_data + 1
		if (word & 0xF00000) == 0xA00000:  # chip header
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
		elif (word & 0xF00000) == 0xB00000:  # chip trailer
			ro_flags = (word & 0x0F0000) >> 16
			if ro_flags != 0x0 :
				print ("Read out flags [0x%x] detected\n" % (ro_flags))
		elif (word & 0xF00000) == 0xE00000:  # chip empty
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
		elif (word & 0xE00000) == 0xC00000:  # reagion header
			region= (word>>16) & 0x1F
		elif (word & 0xC00000) == 0x400000:  # short
			word_shifted = word>>8
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1	#region*32 + encoderid*2 + addr correction
			y = word_shifted>>1 & 0x1FF	#addr/2
			hitmap_matrix[y,x] = hitmap_matrix[y,x] + 1
			n_events = n_events +1
		elif (word & 0xC00000) == 0x000000:  # long
			word_shifted = word>>8
			address = word_shifted & 0x3ff
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1
			y = word_shifted>>1 & 0x1FF
			hitmap_matrix[y,x] = hitmap_matrix[y,x] + 1
			hitmap = word & 0x7F
			n_events = n_events + 1
			for i in range(7):
				hitmap_shifted = hitmap>>i
				if (hitmap_shifted & 0x1) == 0x1:
					addressmap = address + i + 1
					xhm = region<<5 | word_shifted>>9 & 0x1E | (addressmap ^ addressmap>>1) & 0x1	#x hit map
					yhm = addressmap>>1	#y hit map
					hitmap_matrix[yhm, xhm] = hitmap_matrix[yhm, xhm] + 1
					n_events = n_events + 1
		elif word == 0xFFFFFF:	#idle
			n_idle = n_idle + 1
		elif word == 0xF1FFFF:	#busy on
			n_busy = n_busy + 1
		elif word == 0xF0FFFF:	#busy off
			n_busy = n_busy + 1
		#else:
	return n_events, n_idle, n_busy, n_data, region, hitmap_matrix	#returns the hitmap matrix and some statistics


#per event decodification (need to save data into HDD periodically otherwise the RAM will explode)
def per_event_decode(data_block, n_events, n_idle, n_busy, n_data, n_packet, region, packet_list, packet_content):	#data_blok-->raw data
	for word in data_block:
		n_data = n_data + 1
		if (word & 0xF00000) == 0xA00000:  # chip header
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
		elif (word & 0xF00000) == 0xB00000:  # chip trailer
			n_packet = n_packet + 1
			ro_flags = (word & 0x0F0000) >> 16
			packet_list.append(packet_content)
			packet_content = []
			if ro_flags != 0x0 :
				print "Read out flags [0x%x] detected \n" % (ro_flags)
		elif (word & 0xF00000) == 0xE00000:  # chip empty
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
		elif (word & 0xE00000) == 0xC00000:  # reagion header
			region= (word>>16) & 0x1F
		elif (word & 0xC00000) == 0x400000:  # short
			word_shifted = word>>8
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1	#region*32 + encoderid*2 + addr correction
			y = word_shifted>>1 & 0x1FF	#addr/2
			packet_content.append([y,x])
			n_events = n_events +1
		elif (word & 0xC00000) == 0x000000:  # long
			word_shifted = word>>8
			address = word_shifted & 0x3ff
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1
			y = word_shifted>>1 & 0x1FF
			packet_content.append([y,x])
			hitmap = word & 0x7F
			n_events = n_events + 1
			for i in range(7):
				hitmap_shifted = hitmap>>i
				if (hitmap_shifted & 0x1) == 0x1:
					addressmap = address + i + 1
					xhm = region<<5 | word_shifted>>9 & 0x1E | (addressmap ^ addressmap>>1) & 0x1	#x hit map
					yhm = addressmap>>1	#y hit map
					packet_content.append([yhm,xhm])
					n_events = n_events + 1
		elif word == 0xFFFFFF:	#idle
			n_idle = n_idle + 1
		elif word == 0xF1FFFF:	#busy on
			n_busy = n_busy + 1
		elif word == 0xF0FFFF:	#busy off
			n_busy = n_busy + 1
		#else:
	return n_events, n_idle, n_busy, n_data, n_packet, region, packet_list, packet_content	#returns the packets and some statistics


#raw data decodification npy file(rawdata --> txt file (coordinates))
def decode_data(): #decode data and save as txt file DecodedData.txt
	n_events = 0
	rawdata = np.load(os.path.join('Current_run_data/Rawdata_readout', 'Rawdata.npy'))
	file = open("Current_run_data/Rawdata_readout/DecodedData.txt", "w")
	hitmap_matrix  = np.zeros((512,1024))
	for word in rawdata:
		if (word & 0xF00000) == 0xA00000:  # chip header
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
			file.write("CHIP HEADER: plane=%d\n" % plane_id)
			file.write("Bunch counter=%d\n" % bc)
		elif (word & 0xF00000) == 0xB00000:  # chip trailer
			ro_flags = (word & 0x0F0000) >> 16
			file.write("CHIP TRAILER: flags=0x%x\n\n" % ro_flags)
		elif (word & 0xF00000) == 0xE00000:  # chip empty
			plane_id = (word & 0x0F0000) >> 16
			bc = (word & 0x00FF00) >> 8
			file.write("CHIP EMPTY: plane=%d\n" % plane_id)
			file.write("Bunch counter=%d\n\n" % bc)
		elif (word & 0xE00000) == 0xC00000:  # reagion header
			region= (word>>16) & 0x1F
			file.write("REAGION_HEADER: region=%d\n" % region)
		elif (word & 0xC00000) == 0x400000:  # short
			word_shifted = word>>8
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1	#region*32 + encoderid*2 + addr correction
			y = word_shifted>>1 & 0x1FF	#addr/2
			hitmap_matrix[y,x] = hitmap_matrix[y,x] + 1
			file.write("DATA_SHORT: \nx=%d , y=%d \n" % (x, y))
			n_events = n_events +1
		elif (word & 0xC00000) == 0x000000:  # long
			word_shifted = word>>8
			address = word_shifted & 0x3ff
			x = region<<5 | word_shifted>>9 & 0x1E | (word_shifted ^ word_shifted>>1) & 0x1
			y = word_shifted>>1 & 0x1FF
			hitmap_matrix[y,x] = hitmap_matrix[y,x] + 1
			hitmap = word & 0x7F
			file.write("DATA LONG: \nx=%d , y=%d \n" % (x, y))
			n_events = n_events + 1
			for i in range(7):
				hitmap_shifted = hitmap>>i
				if (hitmap_shifted & 0x1) == 0x1:
					addressmap = address + i + 1
					xhm = region<<5 | word_shifted>>9 & 0x1E | (addressmap ^ addressmap>>1) & 0x1	#x hit map
					yhm = addressmap>>1	#y hit map
					hitmap_matrix[yhm, xhm] = hitmap_matrix[yhm, xhm] + 1
					file.write("x=%d , y=%d \n" % (xhm, yhm))
					n_events = n_events + 1
		#elif word == 0xFFFFFF:	#idle
		elif word == 0xF1FFFF:	#busy on
			file.write("BUSY ON\n")
		elif word == 0xF0FFFF:	#busy off
			file.write("BUSY OFF\n")
		# else:
	file.write("Total events detected=%d \n" % n_events)
	file.close()
	print "File saved as DecodedData.txt"

def print_error():
	powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = get_status()
	if err_slave:  # more then 51 clk cycles driven by slave
		print
		"Slave drive on time out"
	if err_idle:
		print
		"high signal not detected on idle phase"
	if err_read:
		print
		"stop bit not detected"
	if err_chip_id:
		print
		"different chip ID recieved"

#It will perform a 120s reading, if a given pixel responds more than 2 times it will be marked as noisy
def mask_noisy_pixels():
	n_events = 0
	n_idle = 0
	n_busy = 0
	n_data = 0
	region = 0
	hitmap_matrix  = np.zeros((512,1024))
	writeregister(0x0001, 0x020E)  # mode cotrol register configuration (see above)
	# FROMU CFG reg 1, enable internal STROBE(bit 3), enable busy monitoring (bit 4)
	writeregister(0x0004, 0x0018)
	writeregister(0x0005, 1000)	#STROBE duration 16 clk cycles (10us)
	writeregister(0x0006, 9600)	#gap between 2 STROBES	384 clk cycles (240us) sample every 250 us-->4kHz
	Start_time=time.time()
	Mask_time=120	#120 seconds
	send_cmd(trigger_cmd)	#need to first external start the STROBE generation
	send_cmd(read_out)
	while ((time.time()-Start_time)<Mask_time):
		mem_readable = hw.getNode("CSR.status.mem_readable").read()	#get memory readable status from FPGA
		hw.dispatch()
		if mem_readable:
			FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
			hw.dispatch()
			n_events, n_idle, n_busy, n_data, region, hitmap_matrix = decode_block(FIFO, n_events, n_idle, n_busy, n_data, region, hitmap_matrix)
			hw.getNode("CSR.ctrl.mem_read").write(0b1)	#send memory read signal to FPGA
			hw.getNode("CSR.ctrl.mem_read").write(0b0)
			hw.dispatch()
	hw.getNode("CSR.ctrl.ro_stop").write(0b1)	#send stop  read out to FPGA
	hw.dispatch()
	writeregister(0x0004, 0x0000)  # FROMU CFG reg 1, disable internal STROBE
	hw.dispatch()
	Masked_pixels=[]
	for row in range(512):
		for column in range (1024):
			if hitmap_matrix[row,column] > 2:
				Masked_pixels.append([row,column])
				maskPixel(column,row, True)
				print "\nMasked pixel x=%d  y=%d\n" % (column, row)
	print "Masked %d Pixels in total" % len(Masked_pixels)
	print_error()
	time.sleep(0.001)
	hw.getNode("CSR.ctrl.ro_stop").write(0b0)
	hw.dispatch()
	return Masked_pixels


def save_hitmap(matrix, file_name):
	fig, ax = plt.subplots()
	colormap = cm.get_cmap('jet')
	psm = ax.pcolormesh(matrix, cmap=colormap, vmin=0, vmax=np.max(matrix))
	cbar=plt.colorbar(psm, shrink=0.5, ax=ax)
	cbar.set_label("N. hits")
	plt.axis('scaled')
	ax.set(xlim=(0, 1023), ylim=(0, 511))
	plt.xlabel("Column")
	plt.ylabel("Row")
	plt.savefig(file_name, dpi=1600)

def save_hitmap_logscale(matrix, file_name):
	fig, ax = plt.subplots()
	colormap = cm.get_cmap('jet')
	psm = ax.pcolormesh(matrix, cmap=colormap, norm=colors.LogNorm(), vmin=1, vmax=np.max(matrix))
	cbar=plt.colorbar(psm, shrink=0.5, ax=ax)
	cbar.set_label("N. hits")
	plt.axis('scaled')
	ax.set(xlim=(0, 1023), ylim=(0, 511))
	plt.xlabel("Column")
	plt.ylabel("Row")
	plt.savefig(file_name, dpi=1600)

#meh, it doesn't work that well
def ADC_calibration():
	writeregister(0x0610, 0x0001)  # calibration mode and AVSS ADC input discriminator = 0
	writeregister(0x0000, 0xFF20)  # mesure command
	time.sleep(0.01)
	val1 = readregister(0x0612)  # store value with dis = 0
	writeregister(0x0610, 0x0101)  # calibration mode and AVSS ADC input discriminator = 1
	writeregister(0x0000, 0xFF20)  # mesure command
	time.sleep(0.01)
	val2 = readregister(0x0612)  # store value with dis = 1
	if val1 > val2 :
		writeregister(0x0610, 0x001D)  # calibration mode and Bandgap ADC input disc=0 HBT=0
		writeregister(0x0000, 0xFF20)  # mesure command
		time.sleep(0.01)
		val3 = readregister(0x0612)  # store value with HBT = 0
		writeregister(0x0610, 0x081D)  # calibration mode and Bandgap ADC input disc=0 HBT=1
		writeregister(0x0000, 0xFF20)  # mesure command
		time.sleep(0.01)
		val4 = readregister(0x0612)  # store value with dis = 1
		if val3 > val4:
			writeregister(0x0610,0x0001)	#cal mode dis=0 HBT=0
			ADC_reg_data = 0x0000
			print 'Discriminator=0 HBT=0\n'
		else :
			writeregister(0x0610, 0x0801)	#cal mode dis=0 HBT=1
			ADC_reg_data = 0x0800
			print 'Discriminator=0 HBT=1\n'
	else :
		writeregister(0x0610, 0x011D)  # calibration mode and Bandgap ADC input disc=1 HBT=0
		writeregister(0x0000, 0xFF20)  # mesure command
		time.sleep(0.01)
		val3 = readregister(0x0612)  # store value with HBT = 0
		writeregister(0x0610, 0x091D)  # calibration mode and Bandgap ADC input disc=1 HBT=1
		writeregister(0x0000, 0xFF20)  # mesure command
		time.sleep(0.01)
		val4 = readregister(0x0612)  # store value with dis = 1
		if val3 > val4:
			writeregister(0x0610, 0x0101)	#cal mode dis=1 HBT=0
			ADC_reg_data = 0x0100
			print 'Discriminator=1 HBT=0\n'
		else :
			writeregister(0x0610, 0x0901)	#cal mode dis=1 HBT=1
			ADC_reg_data = 0x0900
			print 'Discriminator=1 HBT=1\n'
	writeregister(0x0000, 0xFF20)  # mesure command, manual measure on AVSS(0mV)
	time.sleep(0.01)
	offset = readregister(0x0612)
	print "ADC measure offset = %d\n" % offset
	writeregister(0x0610,ADC_reg_data + 2)	#auto mode and whatever dis and HBT
	return offset

def V_out(ADC_VALUE, ADC_OFFSET):
	V_mis = (ADC_VALUE - ADC_OFFSET)*2*1.068 #mV
	return V_mis

def I_out(ADC_VALUE, ADC_OFFSET):
	I_mis = (ADC_VALUE - ADC_OFFSET)*1.068/5 #uA
	return I_mis

#meh, it doesn't work that well
def measure_ADC():
	ADC_offset = ADC_calibration()
	writeregister(0x0000,0xFF20)	#mesure ADC
	time.sleep(0.100)	#wait 100ms
	ADC_VRSTP = readregister(0x61B)
	ADC_VRSTD = readregister(0x61C)
	ADC_AVDD= readregister(0x616)
	ADC_VPULSEH = readregister(0x619)
	ADC_VPULSEL = readregister(0x61A)
	ADC_VCASN = readregister(0x617)
	ADC_VCASP = readregister(0x618)
	ADC_VCASN2 = readregister(0x61D)
	ADC_ITHR = readregister(0x620)
	ADC_TEMP = readregister(0x627)
	VRSTP = V_out(ADC_VRSTP, ADC_offset)
	print "VRSTP=%d mV\n" % (VRSTP - 370)
	VRSTD = V_out(ADC_VRSTD,ADC_offset)
	print "VRSTD=%d mV\n" % (VRSTD-370)
	AVDD = V_out(ADC_AVDD,ADC_offset)
	print "AVDD=%d mV\n" % AVDD
	VPULSEH = V_out(ADC_VPULSEH, ADC_offset)
	print "VPULSEH=%d mV\n" % (VPULSEH-370)
	VPULSEL = V_out(ADC_VPULSEL, ADC_offset)
	print "VPULSEL=%d mV\n" % (VPULSEL-370)
	VCASN = V_out(ADC_VCASN, ADC_offset)
	print "VCASN=%d mV\n" % VCASN
	VCASP = V_out(ADC_VCASP, ADC_offset)
	print "VCASP=%d mV\n" % VCASP
	VCASN2 = V_out(ADC_VCASN2, ADC_offset)
	print "VCASN2=%d mV\n" % VCASN2
	ITHR = I_out(ADC_ITHR, ADC_offset)
	print "(this one should be divided by 54000)ITHR=%d uA\n" % ITHR
	#TEMP_read = (ADC_TEMP - ADC_offset)*0.147-51.5
	TEMP_read = (ADC_TEMP - ADC_offset) * 0.147
	print "Temperature=%d C" % TEMP_read

def maskAllPixels(enable):
	writeregister(0x048F,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	writeregister(0x048F,0xFFFF)
	writeregister(0x0487,0x0000)

def enablePulserAllPixels(enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True  else 0x0001)
	writeregister(0x0487,0xFFFF)
	writeregister(0x0487,0x0000)

def maskRegion(region,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	writeregister((region<<11|0x0403),0xFFFF)
	for i in range(32):
		writeregister((i<<11|0x0404),0xFFFF)
	writeregister(0x0487,0x0000)

def enablePulserRegion(region,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	writeregister((region<<11|0x0403),0xFFFF)
	for i in range(32):
		writeregister((i<<11|0x0404),0xFFFF)
	writeregister(0x0487,0x0000)

#enable pulser on 16 rows (Row_region=0 rows from 0 to 15, Row_region=1 rows from 16 to 31 ...)
def enablePulserRow_Region(Region,nrow,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	for i in range(32):
		writeregister((i<<11|0x0403),0xFFFF)
	writeregister(((Region*16)&0x1F0)<<7|0x0404,nrow)
	writeregister(0x0487,0x0000)

#mask 16 rows (Row_region=0 rows from 0 to 15, Row_region=1 rows from 16 to 31 ...)
def maskRow_Region(Region,nrow,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	for i in range(32):
		writeregister((i<<11|0x0403),0xFFFF)
	writeregister(((Region*16)&0x1F0)<<7|0x0404,nrow)
	writeregister(0x0487,0x0000)

def enablePulserRow(row,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	for i in range(32):
		writeregister((i<<11|0x0403),0xFFFF)
	writeregister((row&0x1F0)<<7|0x0404,1<<(row&0xF))
	writeregister(0x0487,0x0000)

def maskRow(row,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	for i in range(32):
		writeregister((i<<11|0x0403),0xFFFF)
	writeregister((row&0x1F0)<<7|0x0404,1<<(row&0xF))
	writeregister(0x0487,0x0000)

def enablePulserColumn(column,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	writeregister((column&0x3E0)<<6|0x0400|(1+(column>>4&0x1)),1<<(column&0xF))
	for i in range(32):
		writeregister((i<<11|0x0404),0xFFFF)
	writeregister(0x0487,0x0000)

def maskColumn(column,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	writeregister((column&0x3E0)<<6|0x0400|(1+(column>>4&0x1)),1<<(column&0xF))
	for i in range(32):
		writeregister((i<<11|0x0404),0xFFFF)
	writeregister(0x0487,0x0000)

#enable pulser on a specific row of a specific region
def enablePulserRegionRow(region,row,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	writeregister((region<<11|0x0403),0xFFFF)
	writeregister((row&0x1F0)<<7|0x0404,1<<(row&0xF))
	writeregister(0x0487,0x0000)

#mask a specific row of a specific region
def maskRegionRow(region,row,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	writeregister((region<<11|0x0403),0xFFFF)
	writeregister((row&0x1F0)<<7|0x0404,1<<(row&0xF))
	writeregister(0x0487,0x0000)




def maskPixel(x,y,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0002 if enable == True else 0x0000)
	writeregister((x&0x3E0)<<6|0x0400|(1+(x>>4&0x1)),1<<(x&0xF))
	writeregister((y&0x1F0)<<7|0x0404               ,1<<(y&0xF))
	writeregister(0x0487,0x0000)

def enablePulserPixel(x,y,enable):
	writeregister(0x0487,0x0000)
	writeregister(0x0500,0x0003 if enable == True else 0x0001)
	writeregister(((x&0x3E0)<<6)|0x0400|(1+(x>>4&0x1)),1<<(x&0xF))
	writeregister(((y&0x1F0)<<7)|0x0404               ,1<<(y&0xF))
	writeregister(0x0487,0x0000)

def cc_print():
	print """
		ALPIDE COMMAND console
		power on			:	pon
		power off			:	poff
		initialize			:	init
		change Chip_ID 			:	ci
		write broadcast command		:	bc
		write register			:	wr
		multicast write			:	mw
		read register			:	rr
		trigger				:	tr
		clear all error (MASTER)	:	rerr
		readout options			:	roop
		continous read out		:	roc
		per event readout		:	rope
		raw data read out		:	rord
		decode raw data			:	dd
		pulse all pixels		:	pa
		pulse row region		:	prr
		pulse region			:	pr
		pulse row			:	prow
		pulse a pixel			:	ps
		mask all pixels			:	ma
		mask row region			:	mrr
		mask region			:	mr
		mask row			:	mrow
		mask pixel			:	ms
		global threshold test		:	gtt
		single pixel threshod test	:	sptt
		start pulse test		:	spt
		test threshold			:	tt
		print cmd and address		:	pcmd
		list of register addresses	:	lra
		list of cmds			:	lcmd
		list of data			:	ld
		command console			:	cc
		measure				: 	me
		mask noisy pixels		:	mnp
		hard reset			:	reset
	"""
