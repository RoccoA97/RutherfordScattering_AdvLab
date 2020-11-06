create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports {clk_base_xc7a_i}];
create_clock -add -name clk_tx_mac -period 40.00 -waveform {0 20} [get_ports {mii_tx_clk_i}];
create_clock -add -name clk_rx -period 40.00 -waveform {0 20} [get_ports {mii_rx_clk_i}];


set_false_path -from [get_clocks clk_rx] -to [get_clocks clk_tx_mac]
set_false_path -from [get_clocks clk_rx] -to [get_clocks -of_objects [get_pins Inst_system_clocks/MMCME2_ADV_TX_PLL/CLKOUT1]]
set_false_path -from [get_clocks clk_tx_mac] -to [get_clocks -of_objects [get_pins Inst_system_clocks/MMCME2_ADV_TX_PLL/CLKOUT1]]
set_false_path -from [get_clocks -of_objects [get_pins Inst_system_clocks/MMCME2_ADV_TX_PLL/CLKOUT1]] -to [get_clocks clk_rx]
set_false_path -from [get_clocks -of_objects [get_pins Inst_system_clocks/MMCME2_ADV_TX_PLL/CLKOUT1]] -to [get_clocks clk_tx_mac]

set_false_path -through CTRL_STATUS/reg_reg[0][7]/C 
