import uhal
import time
from subprocess import call
import os
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
import argparse
#home-made ALPIDE function
import ALPIDEfunctions as Afunc

matplotlib.rcParams['text.usetex'] = True

parser = argparse.ArgumentParser(description='ALPIDE COMMAND LINE INTERFACE')
parser.add_argument('-cc', '--Commands', metavar='N', type=str, nargs='+', default=['cc'],
                    help='List of commands to perform, type -cc cc to print the cmd list')
parser.add_argument('-ci', '--Chip_ID', metavar='N', type=int,
                    help='Chip ID that will be used, one among 0x12,0x16,0x1A,0x1E, default 0x12')
parser.add_argument('-st', '--STROBE', metavar='N', type=int, nargs='+',
                    help='Strobe duration (first) STROBE gap (second) in clk cycles, default 100')
parser.add_argument('-ti', '--readout_time', metavar='N', type=int, default=1,
                    help='Readout time in minutes, default 1')
parser.add_argument('-pn', '--readout_packets', metavar='N', type=int, default=100000,
                    help='Readout packets, default 100000')
parser.add_argument('-fn', '--file_name', metavar='N', type=str, default='Default_run',
                    help='Name file, same name dir will be created, default-->Default_run')

args = parser.parse_args()

#routine list
power_off = 0b0000
power_on = 0b0001
initialize = 0b0010
read_register = 0b0100
read_out = 0b0101
write_cmd = 0b1000
write_register = 0b1001
trigger_cmd = 0b1010
clear_err = 0b1111

#BROADCAST COMMAND
ALPIDE_CMD_TRIGGER	= 0x0055
ALPIDE_CMD_GRST		= 0x00D2
ALPIDE_CMD_PRST		= 0x00E4
ALPIDE_CMD_PULSE	= 0x0078
ALPIDE_CMD_BCRST	= 0x0036
ALPIDE_CMD_RORST	= 0x0063
ALPIDE_CMD_DEBUG	= 0x00AA
ALPIDE_CMD_WROP		= 0x009C
ALPIDE_CMD_RDOP		= 0x004E

#simple command console to control a single chip ALPIDE via ip_bus (address = 10.10.10.100)
#dead pixels[x,y]= [823,177],[840,162]


if __name__ == "__main__":

	#os.system("rm Current_run_data/Per_event_readout/*")
	#os.system("rm Current_run_data/Continuous_readout/*")
	#os.system("rm Current_run_data/Rawdata_readout/*")
	#os.system("rm Current_run_data/Threshold_test/*")
	Afunc.init()
	global manager
	global hw
	#Connect to the device
	manager = uhal.ConnectionManager("file://ALPIDE_connection.xml")
	hw = manager.getDevice("ALPIDE")
	hw.getNode("CSR.ctrl.strt").write(0b0)	#set the start to '0'
	hw.getNode("CSR.ctrl.rstn_ipBUS").write(0b1)	#set the reset to '1'
	hw.getNode("CSR.ctrl.ro_stop").write(0b0)	#Enable read out (FPGA side)
	hw.dispatch()
	print 'Initialization comlpete'
	
	#get working dir
	path = os.getcwd()
	print ("The current working directory is %s" % path)

	powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()	#get all status
	region = 0
	Masked_pixels_list = []
	RR_addr = 0xFFFF	#register to be read
	OP_cmd = 0xFF	#broadcast command to send (should be a logical restriction among the possibilities)
	Chip_ID = args.Chip_ID	#chip identification one of 0x12,0x16,0x1A,0x1E
	WR_addr = 0xFFFF	#register to be written
	WR_data = 0xFFFF	#data to be written in register
	mode_cfg = 0x020E	#0x020E matrix readout speed every 2 clk cycles and read out form CMU, clustering enabled, continuous mode
	#Set the chip ID
	hw.getNode("cmd_addr.Chip_ID").write(Chip_ID)
	hw.dispatch()
	
		
	
	while len(args.Commands) >= 1:

		ip = args.Commands[0]
		args.Commands.pop(0)
		
		if ip == "pon":
			Afunc.send_cmd(power_on)
			powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()

		elif ip == "poff":
			Afunc.send_cmd(power_off)
			powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()

		elif ip == "init":
			if powered:
				Afunc.send_cmd(initialize)
				powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()
			else:
				print "ALPIDE not powererd"

		elif ip == "ci":
			Chip_ID = input("Write Chip ID, one among 0x12,0x16,0x1A,0x1E or 0x0F for multicast(only write)\n")
			hw.getNode("cmd_addr.Chip_ID").write(Chip_ID)
			hw.dispatch()

		elif ip == "bc":
			if initialized:
				OP_cmd = input("Write the OP command to send\n")
				Afunc.send_broadcast(OP_cmd)
			else:
				print "ALPIDE not initialized"

		elif ip == "wr":
			if initialized:
				WR_addr = input("Write address register\n")
				WR_data = input("Write data to write on address 0x%04x\n" % (WR_addr))
				Afunc.writeregister(WR_addr, WR_data)
				powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()
			else:
				print "ALPIDE not initialized"

		elif ip == "mw":
			if initialized:
				hw.getNode("cmd_addr.Chip_ID").write(0x0F)	#multicast ID
				hw.dispatch()
				WR_addr = input("Write address register\n")
				WR_data = input("Write data to write on address 0x%04x\n" % (WR_addr))
				Afunc.writeregister(WR_addr, WR_data)
				hw.getNode("cmd_addr.Chip_ID").write(Chip_ID)	#reset previous CHIP ID
				hw.dispatch()
			else:
				print "ALPIDE not initialized"

		elif ip == "rr":
			if initialized:
				RR_addr = input("Write address register\n")
				reg_read = Afunc.readregister(RR_addr)
				print "@ Address [0x%04x] is stored the value : 0x%04x" % (RR_addr, reg_read)
				Afunc.print_error()
		elif ip == "tr":
			if initialized:
				Afunc.send_cmd(trigger_cmd)
			else:
				print "ALPIDE not initialized"

		elif ip == "rerr":
			Afunc.send_cmd(clear_err)

		elif ip == "roop":	#read out option
			ro_op_mask = 0x000000
			en_op = input("Select read out mode?(0->Triggered 1->Coninous)\n")
			if en_op == 0:
				ro_op_mask = ro_op_mask | 0x000001  # 1 triggered 2 continous
			elif en_op == 1:
				ro_op_mask = ro_op_mask | 0x000002  # 1 triggered 2 continous
			en_op = input("Enable clustering?(True or False)\n")
			if en_op:
				ro_op_mask = ro_op_mask | 0x000004	#enable on bit 2
			mode_cfg = 0x0208 | ro_op_mask	#0x0208 matrix readout speed every 2 clk cycles and read out form CMU
			Afunc.writeregister(0x0001, mode_cfg)	#mode cotrol register

		elif ip == "roc":
			if initialized:
				folder_path = path + '/Current_run_data/Continuous_readout/' + args.file_name
				try:	#new work folder named args.file_name
				    os.mkdir(folder_path)
				except OSError:
				    print ("Creation of the directory %s failed" % folder_path)
				else:
				    print ("Successfully created the directory %s " % folder_path)
				n_events = 0	#events detected
				n_idle = 0	#data idle recieved
				n_busy = 0 #data busy recieved
				n_data = 0	#data recieved
				STB_dur = args.STROBE[0]
				STB_gap = args.STROBE[1]
				hitmap_matrix  = np.zeros((512,1024))	#hitmap matrix set all to zeros
				Afunc.writeregister(0x0001, mode_cfg)  # mode cotrol register configuration (see above)
				Afunc.writeregister(0x0004, 0x0018)	# FROMU CFG reg 1, enable internal STROBE(bit 3), enable busy monitoring (bit 4)
				Afunc.writeregister(0x0005, STB_dur)	#STROBE duration STB_dur*25ns
				Afunc.writeregister(0x0006, STB_gap)	#gap between 2 STROBES	STB_gap*25ns
				Total_readout_time=time.time()
				Afunc.send_cmd(trigger_cmd)	#need to first external start the STROBE generation
				Afunc.send_cmd(read_out)
				while True:
					try:
						mem_readable = hw.getNode("CSR.status.mem_readable").read()	#get memory readable status from FPGA
						hw.dispatch()
						if mem_readable:
							FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
							hw.dispatch()
							n_events, n_idle, n_busy, n_data, region, hitmap_matrix = Afunc.decode_block(FIFO, n_events, n_idle, n_busy, n_data, region, hitmap_matrix)
							hw.getNode("CSR.ctrl.mem_read").write(0b1)
							hw.getNode("CSR.ctrl.mem_read").write(0b0)
							hw.dispatch()
					except KeyboardInterrupt:
						hw.getNode("CSR.ctrl.ro_stop").write(0b1)
						hw.dispatch()
						Afunc.writeregister(0x0004, 0x0000)  # FROMU CFG reg 1, disable internal STROBE
						Total_readout_time=time.time()-Total_readout_time
						break
				Afunc.print_error()
				dead_time = float(n_idle)/n_data*100
				stat_name_file = 'Stat_' + args.file_name + '.txt'
				file = open(os.path.join(folder_path, stat_name_file), "w")
				file.write('Data recieved	= %d \n' % n_data)
				file.write('Packets recieved = %d \n' % total_packets)
				file.write('Readout_time = %fs \n' % Total_readout_time)
				file.write('Data idle recieved	= %d \n' % n_idle)
				file.write('Data busy recieved	= %d \n' % n_busy)
				file.write('Dead time [percentage] = %f \n' % dead_time)
				file.write('Events detected	= %d \n' % n_events)
				file.write('Throughput[Mbps] = %f \n' % n_data*24/(Total_readout_time*1000000))
				file.close()
				print """
				Data recieved	= %d
				Readout_time = %fs
				Data idle recieved	= %d
				Data busy recieved	= %d
				Dead time [percentage] = %f
				Events detected	= %d
				Throughput[Mbps] = %f
				""" % (n_data, Total_readout_time,  n_idle, n_busy, dead_time, n_events, n_data*24/(Total_readout_time*1000000))
				name_file = args.file_name + '_hitmap_matrix'
				np.save(os.path.join(folder_path, name_file),hitmap_matrix)
				#AAfunc.save_hitmap(hitmap_matrix, 'Hitmap.png')
				#Afunc.save_hitmap_logscale(hitmap_matrix, 'Hitmap_log.png')
				time.sleep(0.001)
				hw.getNode("CSR.ctrl.ro_stop").write(0b0)
				hw.dispatch()
			else:
				print "ALPIDE not initialized"

		elif ip == "rope":	#readout per event
			if initialized:
				folder_path = path + '/Current_run_data/Per_event_readout/' + args.file_name
				try:	#new work folder named args.file_name
				    os.mkdir(folder_path)
				except OSError:
				    print ("Creation of the directory %s failed" % folder_path)
				else:
				    print ("Successfully created the directory %s " % folder_path)
				k = 0 	#save block variable
				Delta_time = 0	#time delta due to saving
				n_events = 0	#events detected
				n_idle = 0	#data idle recieved
				n_busy = 0 #data busy recieved
				n_data = 0	#data recieved
				n_packet = 0	#packets detected (N for each block)
				total_packets = 0	#total number of packet recieved
				packet_list = []	#paket coordinates list
				packet_content = []	#current packet content
				#file = open("Current_run_data/Per_event_readout/Read_out_flags.txt", "w")	#read out falgs will be saved here
				STB_dur = args.STROBE[0]
				STB_gap = args.STROBE[1]
				Afunc.writeregister(0x0001, mode_cfg)  # mode cotrol register configuration (see above)
				Afunc.writeregister(0x0004, 0x0018)	# FROMU CFG reg 1, enable internal STROBE(bit 3), enable busy monitoring (bit 4)
				Afunc.writeregister(0x0005, STB_dur)	#STROBE duration STB_dur*25ns
				Afunc.writeregister(0x0006, STB_gap)	#gap between 2 STROBES	STB_gap*25ns
				Total_readout_time=time.time()
				Afunc.send_cmd(trigger_cmd)	#need to first external start the STROBE generation
				Afunc.send_cmd(read_out)
				while (total_packets < args.pn) and ((time.time()-Total_readout_time-Delta_time) < (args.ti*60)):
					try:
						if n_packet < 100000:
							mem_readable = hw.getNode("CSR.status.mem_readable").read()	#get memory readable status from FPGA
							hw.dispatch()
							if mem_readable:
								FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
								hw.dispatch()
								n_events, n_idle, n_busy, n_data, n_packet, region, packet_list, packet_content = Afunc.per_event_decode(FIFO, n_events, n_idle, n_busy, n_data, n_packet, region, packet_list, packet_content)
								hw.getNode("CSR.ctrl.mem_read").write(0b1)
								hw.getNode("CSR.ctrl.mem_read").write(0b0)
								hw.dispatch()
						else:
							hw.getNode("CSR.ctrl.ro_stop").write(0b1)
							hw.dispatch()
							Afunc.writeregister(0x0004, 0x0000)  # FROMU CFG reg 1, disable internal STROBE
							time_lost=time.time()
							total_packets = total_packets + n_packet
							n_packet = 0
							name_file = args.file_name + '_packet_' + '%d' % (k)
							np.save(os.path.join(folder_path, name_file),packet_list)
							k = k + 1
							packet_list = []
							hw.getNode("CSR.ctrl.ro_stop").write(0b0)
							hw.dispatch()
							time.sleep(0.001)
							Afunc.writeregister(0x0004, 0x0018)	# FROMU CFG reg 1, enable internal STROBE(bit 3), enable busy monitoring (bit 4)
							time_lost=time.time()-time_lost
							Delta_time=Delta_time+time_lost
							Afunc.send_cmd(trigger_cmd)	#need to first external start the STROBE generation
							Afunc.send_cmd(read_out)
					except KeyboardInterrupt:
						break
				hw.getNode("CSR.ctrl.ro_stop").write(0b1)
				hw.dispatch()
				Afunc.writeregister(0x0004, 0x0000)  # FROMU CFG reg 1, disable internal STROBE
				Total_readout_time=time.time()-Total_readout_time-Delta_time
				total_packets = total_packets + n_packet
				n_packet = 0
				name_file = args.file_name + '_packet_' + '%d' % (k)
				np.save(os.path.join(folder_path, name_file),packet_list)
				k = 0
				packet_list = []
				stat_name_file = 'Stat_' + args.file_name + '.txt'
				file = open(os.path.join(folder_path, stat_name_file), "w")
				file.write('Data recieved	= %d \n' % n_data)
				file.write('Packets recieved = %d \n' % total_packets)
				file.write('Readout_time = %fs \n' % Total_readout_time)
				file.write('Data idle recieved	= %d \n' % n_idle)
				file.write('Data busy recieved	= %d \n' % n_busy)
				file.write('Dead time [percentage] = %f \n' % dead_time)
				file.write('Events detected	= %d \n' % n_events)
				file.write('Throughput[Mbps] = %f \n' % n_data*24/(Total_readout_time*1000000))
				file.close()	
				Afunc.print_error()
				dead_time = float(n_idle)/n_data*100
				print """
				Data recieved	= %d
				Packets recieved = %d
				Readout_time = %fs
				Data idle recieved	= %d
				Data busy recieved	= %d
				Dead time [percentage] = %f
				Events detected	= %d
				Throughput[Mbps] = %f
				""" % (n_data, total_packets, Total_readout_time, n_idle, n_busy, dead_time, n_events, n_data*24/(Total_readout_time*1000000))
				#file.close()	#close Read out flags txt file
				#AAfunc.save_hitmap(hitmap_matrix, 'Hitmap.png')
				#Afunc.save_hitmap_logscale(hitmap_matrix, 'Hitmap_log.png')
				time.sleep(0.001)
				hw.getNode("CSR.ctrl.ro_stop").write(0b0)
				hw.dispatch()
			else:
				print "ALPIDE not initialized"

		elif ip == "rord":	#raw data readout
			if initialized:
				raw_data=[]
				Afunc.writeregister(0x0001, mode_cfg)  # mode cotrol register configuration (see above)
				# FROMU CFG reg 1, enable internal STROBE(bit 3), enable busy monitoring (bit 4)
				Afunc.writeregister(0x0004, 0x0018)
				Afunc.writeregister(0x0005, 0x000F)	#STROBE duration 4 clk cycles (400ns)
				Afunc.writeregister(0x0006, 0x0640)	#gap between 2 STROBES	640 clk cycles (16us)
				Afunc.send_cmd(trigger_cmd)	#need to first external start the STROBE generation
				Afunc.send_cmd(read_out)
				while True:
					try:
						mem_readable = hw.getNode("CSR.status.mem_readable").read()	#get memory readable status from FPGA
						hw.dispatch()
						if mem_readable:
							FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
							hw.dispatch()
							raw_data.extend(FIFO)
							hw.getNode("CSR.ctrl.mem_read").write(0b1)	#send memory read signal to FPGA
							hw.getNode("CSR.ctrl.mem_read").write(0b0)
							hw.dispatch()
					except KeyboardInterrupt:
						hw.getNode("CSR.ctrl.ro_stop").write(0b1)	#send stop  read out to FPGA
						hw.dispatch()
						Afunc.writeregister(0x0004, 0x0000)  # FROMU CFG reg 1, disable internal STROBE
						np.save(os.path.join('Current_run_data/Rawdata_readout', 'Rawdata'), raw_data)
						raw_data = []
						break
				Afunc.print_error()
				time.sleep(0.001)
				hw.getNode("CSR.ctrl.ro_stop").write(0b0)
				hw.dispatch()
			else:
				print "ALPIDE not initialized"
		elif ip == "dd":
			Afunc.decode_data()
		elif ip == "pa":
			en = input("Enable Pulser on all pixels? (True or False)\n")
			Afunc.enablePulserAllPixels(en)
		elif ip == "prr":
			rowregion = input("Row Region\n")
			nrow = input("Wich rows\n")
			en = input("Enable or disable pulse on selected region(True or False)\n")
			Afunc.enablePulserRow_Region(rowregion,nrow,en)
		elif ip == "pr":
			region = input("Region\n")
			en = input("Enable or disable pulse on selected region(True or False)\n")
			Afunc.enablePulserRegion(region,en)
		elif ip == "prow":
			row = input("Row\n")
			en = input("Enable or disable pulse on selected region(True or False)\n")
			Afunc.enablePulserRow(row,en)
		elif ip == "ps":
			x = input("pixel x coordinate\n")
			y = input("pixel y coordinate\n")
			en = input("Enable or disable pulse on selected pixel(True or False)\n")
			Afunc.enablePulserPixel(x,y,en)
		elif ip == "spt":	#start pulse test
			raw_data = []
			Afunc.writeregister(0x0606, 0x0000)  # Set PULSEL to 0.37V (DAC offset)
			Afunc.writeregister(0x0605, 0x00ff)  # Set PULSEH to 1.8V
			Pt = raw_input("Pulse type Analog(a) or Digital(d)\n")
			if Pt == "a":
				Afunc.writeregister(0x0004,0x0060)	#Analog pulse, automatic strobe after pulse command
			elif Pt == "d":
				Afunc.writeregister(0x0004,0x0040)  # Digital pulse, automatic strobe after pulse command
			else:
				print "Bad input"
				break
			Afunc.writeregister(0x0001, 0x020E)  # mode cotrol register configuration
			Afunc.writeregister(0x0005, 0x0002)  # STROBE duration (n+1) clk clycles
			Afunc.writeregister(0x0006, 0x0000)  # gap between 2 STROBES (n+1) clk clycles
			Afunc.writeregister(0x0007, 0x0013)  # delay from PULSE to STROBE (n+1) clk cycles
			Afunc.writeregister(0x0008, 0x0016)  # PULSE duration n clk cylces
			time.sleep(0.001)
			Afunc.send_broadcast(0x78)	#send PULSE command
			Afunc.send_cmd(read_out)
			while True:
				try:
					mem_readable = hw.getNode("CSR.status.mem_readable").read()
					hw.dispatch()
					if mem_readable:
						FIFO = hw.getNode("DATA.ro_data").readBlock(0x200)
						hw.dispatch()
						raw_data.extend(FIFO)
						hw.getNode("CSR.ctrl.mem_read").write(0b1)
						hw.getNode("CSR.ctrl.mem_read").write(0b0)
						hw.dispatch()
				except KeyboardInterrupt:
					hw.getNode("CSR.ctrl.ro_stop").write(0b1)
					hw.dispatch()
					np.save('Rawdata', raw_data)  # save data on npy file
					raw_data = []
					break
			Afunc.writeregister(0x0004, 0x0000)
			Afunc.writeregister(0x0007, 0x0000)
			Afunc.writeregister(0x0008, 0x0000)
			powered, initialized, busy, FIFO_full, FIFO_empty, FIFO_prg_full, mem_readable, err_slave, err_idle, err_read, err_chip_id, lkd_AC, lkd_ipbus = Afunc.get_status()
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
		elif ip == "sptt":
			x = input("pixel x coordinate\n")
			y = input("pixel y coordinate\n")
			Afunc.SP_thr_test(x,y)
		elif ip == "tt":  # start test threshold
			file = open("Threshold_test.txt", "w")
			Afunc.enablePulserAllPixels(False)	#disable pulse on all pixels
			Afunc.maskAllPixels(True)	#mask all pixels
			for k in range(20):
				for i in range(10):
					Afunc.maskPixel(30+k, 485+i, False)	#unmask selected pixel
					Afunc.enablePulserPixel(30+k, 485+i, True) #enable pulse on 200 pixels
			Afunc.writeregister(0x0004, 0x0060)  # Analog pulse, automatic strobe after pulse command
			Afunc.writeregister(0x0606, 0x0000)  # Set PULSEL to 0.37V (DAC offset)
			for k in range (120):
				pulse_test_array = []  # initialize empty array
				Q_inj = ((k)*7.06e-3*230e-18)/(1.602e-19)
				Afunc.writeregister(0x0605,k)  # Set PULSEH to k*7.06 mV
				time.sleep(0.0001)
				for j in range (15):
					time.sleep(0.0001)
					ppf = Afunc.pulse_test(mode_cfg) #pixel pulsed fraction
					pulse_test_array.append(ppf)
					#np.append(pulse_test_array, ppf)
				mean = np.mean(pulse_test_array)
				std = np.nanstd(pulse_test_array)
				file.write("%f	%f	0	%f\n" % (Q_inj, mean, std))
			file.close()
			Afunc.writeregister(0x0004, 0x0000)
			Afunc.maskAllPixels(False)	#unmask all pixels
		elif ip == "ma":
			en = input("Mask or unmask all pixels? (True for masking)\n")
			Afunc.maskAllPixels(en)
		elif ip == "mr":
			region = input("Region\n")
			en = input("Mask or unmask(True or False)\n")
			Afunc.maskRegion(region,en)
		elif ip == "mrr":
			rowregion = input("Row Region\n")
			nrow = input("Wich rows\n")
			en = input("Mask or unmask(True or False)\n")
			Afunc.maskRow_Region(rowregion,nrow,en)
		elif ip == "mrow":
			row = input("Row\n")
			en = input("Mask or unmask(True or False)\n")
			Afunc.maskRow(row,en)
		elif ip == "ms":
			x = input("pixel x coordinate\n")
			y = input("pixel y coordinate\n")
			en = input("Mask or unmask(True or False)\n")
			Afunc.maskPixel(x,y,en)
		elif ip == "pcmd":
			RR_addr = hw.getNode("cmd_addr.RR_addr").read()
			OP_command = hw.getNode("cmd_addr.OP_command").read()
			Chip_ID = hw.getNode("cmd_addr.Chip_ID").read()
			WR_addr = hw.getNode("cmd_addr.WR_addr").read()
			WR_data = hw.getNode("cmd_addr.WR_data").read()
			hw.dispatch()
			print """
				RR_addr	= 0x%04x
				OP_cmd	= 0x%02x
				Chip_ID	= 0x%02x
				WR_addr	= 0x%04x
				WR_data	= 0x%04x
				""" % (RR_addr, OP_command, Chip_ID, WR_addr, WR_data)
		elif ip == "lra":
			print """
				ALPIDE_REG_COMMAND	= 0x0000
				ALPIDE_REG_MODE_CTRL	= 0x0001
				ALPIDE_REG_FROMU_CFG1	= 0x0004
				ALPIDE_REG_FROMU_CFG2	= 0x0005
				ALPIDE_REG_FROMU_CFG3	= 0x0006
				ALPIDE_REG_CMUDMU_CFG	= 0x0010
				ALPIDE_REG_DMU_FIFO_LOW	= 0x0012
				ALPIDE_REG_DMU_FIFO_HI	= 0x0013
				ALPIDE_REG_DAC_VRESETD	= 0x0602
				ALPIDE_REG_DAC_VCASN	= 0x0604
				ALPIDE_REG_DAC_VPULSEH	= 0x0605
				ALPIDE_REG_DAC_VPULSEL	= 0x0606
				ALPIDE_REG_DAC_VCASN2	= 0x0607
				ALPIDE_REG_DAC_ITHR	= 0x060E
			"""
		elif ip == "lcmd":
			print """
				ALPIDE_CMD_TRIGGER	= 0x0055
				ALPIDE_CMD_GRST		= 0x00D2
				ALPIDE_CMD_PRST		= 0x00E4
				ALPIDE_CMD_PULSE	= 0x0078
				ALPIDE_CMD_BCRST	= 0x0036
				ALPIDE_CMD_RORST	= 0x0063
				ALPIDE_CMD_DEBUG	= 0x00AA
				ALPIDE_CMD_WROP		= 0x009C
				ALPIDE_CMD_RDOP		= 0x004E
				ALPIDE_CMD_CMU_CLEAR_ERR= 0xFF00
				ALPIDE_CMD_FIFOTEST	= 0xFF01
				ALPIDE_CMD_LOADOBDEFCFG	= 0xFF02
				ALPIDE_CMD_XOFF		= 0xFF10
				ALPIDE_CMD_XON		= 0xFF11
				ALPIDE_CMD_ADCMEASURE	= 0xFF20
			"""
		elif ip == "ld":
			print """
				ALPIDE_DATA_CHIP_HEADER[16]	= 0xA(ID[4])(BC[8])
				ALPIDE_DATA_CHIP_TRAILER[8]	= 0xB(ID[4])
				ALPIDE_DATA_CHIP_EMPTY[16]	= 0xE(ID[4])(BC[8])
				ALPIDE_DATA_REGION_HEADER[8]	= 0b110(RH[5])
				ALPIDE_DATA_DATA_SHORT[16]	= 0b01(ENID[4])(addr[10])
				ALPIDE_DATA_DATA_LONG[24]	= 0b00(ENID[4])(addr[10])0(HITMAP[7])
				ALPIDE_DATA_BUSY_ON[8]		= 0xF1
				ALPIDE_DATA_BUSY_OFF[8]		= 0xF0
			"""
		elif ip == "cc":
			Afunc.cc_print()
		elif ip == "me":
			Afunc.measure_ADC()
		elif ip == "mnp":
			if initialized:
				Masked_pixels_list = Afunc.mask_noisy_pixels()
			else:
				print "ALPIDE not initialized"

		elif ip == "gtt":
			Afunc.global_thr_test(50,Masked_pixels_list)

		elif ip == "reset":
			#perform an hard reset
			hw.getNode("CSR.ctrl.rstn_ipBUS").write(0b0)
			hw.dispatch()
			time.sleep(1)
			hw.getNode("CSR.ctrl.ro_stop").write(0b0)
			hw.getNode("CSR.ctrl.rstn_ipBUS").write(0b1)
			hw.dispatch()
			print 'Reset completed'

		else:
			print "Command not found\n"

	#Afunc.send_cmd(power_off)
