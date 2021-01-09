#!/usr/bin/expect

# make executable: chmod 755 test.sh
# usage:           ./test.sh -d pass1 pass2
set timeout 60
set pass1 [lindex $argv 1];
set pass2 [lindex $argv 2];

spawn ssh -X ardino@gate.pd.infn.it

expect "*: "
send "$pass1\r"

sleep 1
send "\r"
sleep 1
expect "*\$ "
send "ssh -X collazuo@lxapix01.fisica.unipd.it\r"
expect "*: "
send "$pass2\r"
sleep 1

# expect "*\$ "
send "cd /home/collazuo/Scrivania/RuShield_Control/v3/main\r"
# send "./main prova_expect.script\r"
send "exit\r"
sleep 1
send "exit\r"
interact
