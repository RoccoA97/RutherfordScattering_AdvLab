#!/usr/bin/expect

# make executable: chmod 755 test.sh
# usage:           ./test.sh -d pass1 pass2
# set timeout 60
# set freq [lindex $argv 1];
# set thld [lindex $argv 2];
# set time [lindex $argv 3];
#
# # hard reset
# expect "*\$ "
# send "python ALPIDE_CLI.py -cc pon init reset bc\r"
# sleep 1
# send "0x63\r"
# sleep 1
#
# # initialize
# expect "*\$ "
# send "python ALPIDE_CLI.py -cc init\r"
# sleep 1
#
# expect "*\$ "
# send "python ALPIDE_CLI.py -cc wr\r"
# sleep 1
# send "0x1\r"
# sleep 1
# send "$freq\r" # set to 10 MHz
# sleep 1
#
# expect "*\$ "
# send "python ALPIDE_CLI.py -cc wr\r"
# sleep 1
# send "0x0604\r"
# sleep 1
# send "$thld\r" # set to 10 MHz
# sleep 1
#
# # start acquisition
# expect "*\$ "
# send "python ALPIDE_CLI.py -cc rope -ti $time -pn 100000000 -fn data_VCASN_$thld -st 5000 50000\r"
# interact
set timeout 60
sleep 1
expect "*"
spawn python test.py
sleep 1
expect "*"
# expect -d -re {Prova\n} # -re "(.*)\n"
send "24\r"
sleep 1
# expect "24" { send_user "100\r" }
# send "100\r"
expect eof
spawn pwd
sleep 1
expect "*"
send "pwd\r"
interact
