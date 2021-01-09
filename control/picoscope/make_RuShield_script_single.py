import argparse

parser = argparse.ArgumentParser(description="Make RuShield acquisition script")

parser.add_argument("-s",  "--step",      type=int, help="Angular position in number of steps")
parser.add_argument('-th', "--timeH",     type=int, help="Acquisition time hours")
parser.add_argument('-tm', "--timeM",     type=int, help="Acquisition time minutes")
parser.add_argument('-ts', "--timeS",     type=int, help="Acquisition time seconds")
parser.add_argument('-nc', "--ncaptures", type=int, help="Maximum number of events to acquire")
parser.add_argument('-ns', "--nsegments", type=int, help="Number of segments in which Picoscope memory is divided")

args = parser.parse_args()


acqs = [
    # step  run_time    n_captures  n_segments
    [ args.step, [args.timeH, args.timeM, args.timeS], args.ncaptures, args.nsegments]
]



script  = ""
script += "CalibrateSource;\n"
script += "PowerUp;\n"
script += "CenterSource;\n"
script += "CHANNEL_NAME_combo A;\n"
script += "CHANNEL_RANGE_combo 5V;\n"
script += "RESOLUTION_combo 12bit;\n"
script += "SAMPLESDIV_spin 500;\n"
script += "SRMULTIPLIER_combo MS/s;\n"
script += "SAMPLERATE_spin 200;\n"
script += "TRIGGER_SOURCE_combo A;\n"
script += "TRIGGER_text 500;\n"
script += "TRIGGER_DIRECTION_combo Falling;\n"
script += "PRETRIGGER_text 40;\n"

for acq in acqs:
    script += "CenterSource;\n"
    if (acq[0] < 0):
        script += "MOTOR_DIRECTION_radio 0;\n"
    else:
        script += "MOTOR_DIRECTION_radio 1;\n"
    script += "ChangeDIR;\n"
    script += "STEP_ANGLE_text " + str(acq[0]*0.9) + ";\n"
    script += "RotateSource;\n"
    script += "RUN_TIME_1_spin " + str(acq[1][0]) + ";\n"
    script += "RUN_TIME_2_spin " + str(acq[1][1]) + ";\n"
    script += "RUN_TIME_3_spin " + str(acq[1][2]) + ";\n"
    script += "N_CAPTURES_spin " + str(acq[2]) + ";\n"
    script += "N_SEGMENTS_spin " + str(acq[3]) + ";\n"
    if (acq[0] < 0):
        script += "OUTPUT_PATH_text /home/collazuo/Scrivania/Rutherford_2020/" + "TODO" + str(acq[0]) + "_SX;\n"
    else:
        script += "OUTPUT_PATH_text /home/collazuo/Scrivania/Rutherford_2020/" + "TODO" + str(acq[0]) + "_DX;\n"
    script += "SetPath;\n"
    script += "PicoScopeStart;\n"

script += "PowerDOWN;\n"
script += "OnClose;"

print(script)
