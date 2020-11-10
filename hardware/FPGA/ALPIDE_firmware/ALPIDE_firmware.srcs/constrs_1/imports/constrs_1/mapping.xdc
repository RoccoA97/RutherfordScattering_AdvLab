#set_property SEVERITY {Warning} [get_drc_checks NSTD-1]

set_property PACKAGE_PIN E3 [get_ports {clk_base_xc7a_i}]
set_property IOSTANDARD LVCMOS33 [get_ports {clk_base_xc7a_i}]

set_property PACKAGE_PIN H16 [get_ports {mii_tx_clk_i}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_tx_clk_i}]
set_property PACKAGE_PIN F15 [get_ports {mii_rx_clk_i}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rx_clk_i}]

set_property PACKAGE_PIN H14 [get_ports {mii_txd_o[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_txd_o[0]}]
set_property PACKAGE_PIN J14 [get_ports {mii_txd_o[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_txd_o[1]}]
set_property PACKAGE_PIN J13 [get_ports {mii_txd_o[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_txd_o[2]}]
set_property PACKAGE_PIN H17 [get_ports {mii_txd_o[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_txd_o[3]}]

set_property PACKAGE_PIN H15 [get_ports {mii_tx_en_o}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_tx_en_o}]
set_property PACKAGE_PIN G16 [get_ports {mii_rx_dv_i}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rx_dv_i}]

set_property PACKAGE_PIN D18 [get_ports {mii_rxd_i[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rxd_i[0]}]
set_property PACKAGE_PIN E17 [get_ports {mii_rxd_i[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rxd_i[1]}]
set_property PACKAGE_PIN E18 [get_ports {mii_rxd_i[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rxd_i[2]}]
set_property PACKAGE_PIN G17 [get_ports {mii_rxd_i[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {mii_rxd_i[3]}]

set_property PACKAGE_PIN G18 [get_ports {clk_phy_o}]
set_property IOSTANDARD LVCMOS33 [get_ports {clk_phy_o}]
set_property PACKAGE_PIN C16 [get_ports {rst_n_phy_o}]
set_property IOSTANDARD LVCMOS33 [get_ports {rst_n_phy_o}]

set_property PACKAGE_PIN T10 [get_ports {xc7a_led0_o}]
set_property IOSTANDARD LVCMOS33 [get_ports {xc7a_led0_o}]
set_property PACKAGE_PIN B8 [get_ports {xc7a_rst_i}]
set_property IOSTANDARD LVCMOS33 [get_ports {xc7a_rst_i}]

#ALPIDE
## Switches
set_property -dict { PACKAGE_PIN A8    IOSTANDARD LVCMOS33 } [get_ports { op_sw_i[0] }]; #IO_L12N_T1_MRCC_16 Sch=sw[0]
set_property -dict { PACKAGE_PIN C11   IOSTANDARD LVCMOS33 } [get_ports { op_sw_i[1] }]; #IO_L13P_T2_MRCC_16 Sch=sw[1]
set_property -dict { PACKAGE_PIN C10   IOSTANDARD LVCMOS33 } [get_ports { op_sw_i[2] }]; #IO_L13N_T2_MRCC_16 Sch=sw[2]
set_property -dict { PACKAGE_PIN A10   IOSTANDARD LVCMOS33 } [get_ports { op_sw_i[3] }]; #IO_L14P_T2_SRCC_16 Sch=sw[3]

## RGB LEDs
#set_property -dict { PACKAGE_PIN E1    IOSTANDARD LVCMOS33 } [get_ports { led0_b}]; #IO_L18N_T2_35 Sch=led0_b
#set_property -dict { PACKAGE_PIN F6    IOSTANDARD LVCMOS33 } [get_ports { led0_g }]; #IO_L19N_T3_VREF_35 Sch=led0_g
set_property -dict { PACKAGE_PIN G6    IOSTANDARD LVCMOS33 } [get_ports { id_err_led_o }]; #IO_L19P_T3_35 Sch=led0_r
#set_property -dict { PACKAGE_PIN G4    IOSTANDARD LVCMOS33 } [get_ports { led1_b }]; #IO_L20P_T3_35 Sch=led1_b
#set_property -dict { PACKAGE_PIN J4    IOSTANDARD LVCMOS33 } [get_ports { led1_g }]; #IO_L21P_T3_DQS_35 Sch=led1_g
set_property -dict { PACKAGE_PIN G3    IOSTANDARD LVCMOS33 } [get_ports { read_err_led_o }]; #IO_L20N_T3_35 Sch=led1_r
#set_property -dict { PACKAGE_PIN H4    IOSTANDARD LVCMOS33 } [get_ports { led2_b }]; #IO_L21N_T3_DQS_35 Sch=led2_b
#set_property -dict { PACKAGE_PIN J2    IOSTANDARD LVCMOS33 } [get_ports { led2_g }]; #IO_L22N_T3_35 Sch=led2_g
set_property -dict { PACKAGE_PIN J3    IOSTANDARD LVCMOS33 } [get_ports { idle_err_led_o }]; #IO_L22P_T3_35 Sch=led2_r
#set_property -dict { PACKAGE_PIN K2    IOSTANDARD LVCMOS33 } [get_ports { led3_b}]; #IO_L23P_T3_35 Sch=led3_b
#set_property -dict { PACKAGE_PIN H6    IOSTANDARD LVCMOS33 } [get_ports { led3_g }]; #IO_L24P_T3_35 Sch=led3_g
set_property -dict { PACKAGE_PIN K1    IOSTANDARD LVCMOS33 } [get_ports { slave_err_led_o }]; #IO_L23N_T3_35 Sch=led3_r

## LEDs
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports { power_led_o }]; #IO_L24N_T3_35 Sch=led[4]
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports { init_led_o }]; #IO_25_35 Sch=led[5]
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports { busy_led_o }]; #IO_L24P_T3_A01_D17_14 Sch=led[6]
#set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]; #IO_L24N_T3_A00_D16_14 Sch=led[7]

## Buttons
set_property -dict { PACKAGE_PIN D9    IOSTANDARD LVCMOS33 } [get_ports { btn_i }]; #IO_L6N_T0_VREF_16 Sch=btn[0]
set_property -dict { PACKAGE_PIN C9    IOSTANDARD LVCMOS33 } [get_ports { wcmd_enable_i }]; #IO_L11P_T1_SRCC_16 Sch=btn[1]
#set_property -dict { PACKAGE_PIN B9    IOSTANDARD LVCMOS33 } [get_ports { btn[2] }]; #IO_L11N_T1_SRCC_16 Sch=btn[2]

## Pmod Header JA
#set_property -dict { PACKAGE_PIN G13   IOSTANDARD LVCMOS33 } [get_ports { CTRL_io }]; #IO_0_15 Sch=ja[1]
#set_property -dict { PACKAGE_PIN B11   IOSTANDARD LVDS_25 } [get_ports { sig_p }]; #IO_L4P_T0_15 Sch=ja[2]
#set_property -dict { PACKAGE_PIN A11   IOSTANDARD LVDS_25 } [get_ports { sig_n }]; #IO_L4N_T0_15 Sch=ja[3]
#set_property -dict { PACKAGE_PIN D12   IOSTANDARD LVCMOS33 } [get_ports { CTRL_io }]; #IO_L6P_T0_15 Sch=ja[4]
#set_property -dict { PACKAGE_PIN D13   IOSTANDARD LVCMOS33 } [get_ports { ja[4] }]; #IO_L6N_T0_VREF_15 Sch=ja[7]
#set_property -dict { PACKAGE_PIN B18   IOSTANDARD LVCMOS33 } [get_ports { ja[5] }]; #IO_L10P_T1_AD11P_15 Sch=ja[8]
#set_property -dict { PACKAGE_PIN A18   IOSTANDARD LVCMOS33 } [get_ports { ja[6] }]; #IO_L10N_T1_AD11N_15 Sch=ja[9]
#set_property -dict { PACKAGE_PIN K16   IOSTANDARD LVCMOS33 } [get_ports { DCLK_o }]; #IO_25_15 Sch=ja[10]

## Pmod Header JB
#set_property -dict { PACKAGE_PIN E15   IOSTANDARD LVCMOS33 } [get_ports { jb[0] }]; #IO_L11P_T1_SRCC_15 Sch=jb_p[1]
set_property -dict { PACKAGE_PIN E16   IOSTANDARD LVCMOS33 } [get_ports { pwr_enable_o }]; #IO_L11N_T1_SRCC_15 Sch=jb_n[1]
#set_property -dict { PACKAGE_PIN D15   IOSTANDARD LVCMOS33 } [get_ports { jb[2] }]; #IO_L12P_T1_MRCC_15 Sch=jb_p[2]
set_property -dict { PACKAGE_PIN C15   IOSTANDARD LVCMOS33 } [get_ports { CTRL_io }]; #IO_L12N_T1_MRCC_15 Sch=jb_n[2]
set_property -dict { PACKAGE_PIN J17   IOSTANDARD LVCMOS33 } [get_ports { DCLK_o }]; #IO_L23P_T3_FOE_B_15 Sch=jb_p[3]
#set_property -dict { PACKAGE_PIN J18   IOSTANDARD LVCMOS33 } [get_ports { jb[5] }]; #IO_L23N_T3_FWE_B_15 Sch=jb_n[3]
set_property -dict { PACKAGE_PIN K15   IOSTANDARD LVCMOS33 } [get_ports { rst_n_io_o }]; #IO_L24P_T3_RS1_15 Sch=jb_p[4]
#set_property -dict { PACKAGE_PIN J15   IOSTANDARD LVCMOS33 } [get_ports { jb[7]}]; #IO_L24N_T3_RS0_15 Sch=jb_n[4]

## ChipKit Outer Digital Header
#set_property -dict { PACKAGE_PIN V15   IOSTANDARD LVCMOS33 } [get_ports { ck_io0  }]; #IO_L16P_T2_CSI_B_14          Sch=ck_io[0]
#set_property -dict { PACKAGE_PIN U16   IOSTANDARD LVCMOS33 } [get_ports { ck_io1  }]; #IO_L18P_T2_A12_D28_14        Sch=ck_io[1]
#set_property -dict { PACKAGE_PIN P14   IOSTANDARD LVCMOS33 } [get_ports { ck_io2  }]; #IO_L8N_T1_D12_14             Sch=ck_io[2]
#set_property -dict { PACKAGE_PIN T11   IOSTANDARD LVCMOS33 } [get_ports { ck_io3  }]; #IO_L19P_T3_A10_D26_14        Sch=ck_io[3]
#set_property -dict { PACKAGE_PIN R12   IOSTANDARD LVCMOS33 } [get_ports { ck_io4  }]; #IO_L5P_T0_D06_14             Sch=ck_io[4]
#set_property -dict { PACKAGE_PIN T14   IOSTANDARD LVCMOS33 } [get_ports { ck_io5  }]; #IO_L14P_T2_SRCC_14           Sch=ck_io[5]
#set_property -dict { PACKAGE_PIN T15   IOSTANDARD LVCMOS33 } [get_ports { ck_io6  }]; #IO_L14N_T2_SRCC_14           Sch=ck_io[6]
#set_property -dict { PACKAGE_PIN T16   IOSTANDARD LVCMOS33 } [get_ports { rst_n_io_o  }]; #IO_L15N_T2_DQS_DOUT_CSO_B_14 Sch=ck_io[7]
#set_property -dict { PACKAGE_PIN N15   IOSTANDARD LVCMOS33 } [get_ports { CTRL_io  }]; #IO_L11P_T1_SRCC_14           Sch=ck_io[8]
#set_property -dict { PACKAGE_PIN M16   IOSTANDARD LVCMOS33 } [get_ports { pwr_enable_o  }]; #IO_L10P_T1_D14_14            Sch=ck_io[9]
#set_property -dict { PACKAGE_PIN V17   IOSTANDARD LVCMOS33 } [get_ports { DCLK_o }]; #IO_L18N_T2_A11_D27_14        Sch=ck_io[10]
#set_property -dict { PACKAGE_PIN U18   IOSTANDARD LVCMOS33 } [get_ports { ck_io11 }]; #IO_L17N_T2_A13_D29_14        Sch=ck_io[11]
#set_property -dict { PACKAGE_PIN R17   IOSTANDARD LVCMOS33 } [get_ports { ck_io12 }]; #IO_L12N_T1_MRCC_14           Sch=ck_io[12]
#set_property -dict { PACKAGE_PIN P17   IOSTANDARD LVCMOS33 } [get_ports { ck_io13 }]; #IO_L12P_T1_MRCC_14           Sch=ck_io[13]


## Misc. ChipKit Ports
#set_property -dict { PACKAGE_PIN M17   IOSTANDARD LVCMOS33 } [get_ports { ck_ioa }]; #IO_L10N_T1_D15_14 Sch=ck_ioa
set_property -dict { PACKAGE_PIN C2    IOSTANDARD LVCMOS33 } [get_ports { rstn_i }]; #IO_L16P_T2_35 Sch=ck_rst